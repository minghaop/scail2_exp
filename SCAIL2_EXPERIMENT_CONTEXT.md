# SCAIL2 推理吞吐量实验交接文档

更新时间：2026-08-27

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

### 当前实验资源与数据约束（2026-08-27）

- 实验直接在宿主机 conda 环境 `scail2-single-gpu` 中运行，不通过容器。
- 实验模型固定从本地存储 `/raid/scail-2-20260819` 读取。
- 自 2026-08-27 最新指令起，单卡实验只允许使用物理 GPU 2，即 `CUDA_VISIBLE_DEVICES=2`。
- 自 2026-08-27 最新指令起，双卡实验只允许使用物理 GPU 2、3，即 `CUDA_VISIBLE_DEVICES=2,3`；此前使用 GPU 4、5 的记录仅为历史实验事实。
- 其他 GPU 不得用于实验任务。
- 当前测试数据为 `testdata/101`；常规实验固定使用这一组，只有明确需要比较输入时才切换。
- 无服务双卡 FSDP 命令行入口为 `run_fsdp_experiment.py`。它固定绑定物理 GPU 2、3，并复用当前 `Scail2InferenceEngine`、T5 FSDP 和 DiT FSDP 实现。
- `run_fsdp_experiment.py` 默认将每次运行的完整 stdout/stderr 实时写入 `experiment_logs/fsdp_baseline/<输出视频文件名>.log`，同时保留终端输出；落盘的每条日志均带有本地时区和毫秒精度的 ISO 8601 时间戳。

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

## 14. 双卡 FSDP init-only 基线（2026-08-27）

- 命令：`python -u run_fsdp_experiment.py --init-only`
- GPU：物理 GPU 4、5；初始化结束后两卡显存均释放到 0 MiB。
- 日志：`experiment_logs/fsdp_baseline/101-20260827-133218.log`
- 结果：成功完成 process group、全部模型加载、DiT FSDP 包装和 readiness 同步后退出；没有进入推理，也没有生成 MP4。
- 两个 rank 的 `engine_load` 均为 289.269 秒，`pipeline_load` 约为 284.9 秒。
- 关键 rank（rank 0）的主要阶段：T5 加载/FSDP 79.668 秒，VAE 2.537 秒，CLIP 1.417 秒，DiT 空模型构建及 BF16 转换 178.546 秒，DiT checkpoint 复制 5.728 秒，DiT FSDP 包装及同步 11.790 秒。
- rank 1 的 T5 阶段更快（65.696 秒），因此在 `pre_fsdp_barrier` 等待 rank 0 共 11.229 秒；rank 0 只等待 0.283 秒。当前双卡初始化关键路径由 rank 0 主导。
- `warmup()` 当前只执行 CUDA synchronize 和 distributed barrier，不进行模型 forward；本基线衡量的是模型可用前的加载/同步成本，而非首个推理请求的 kernel/JIT 预热成本。

## 15. DiT meta+assign 初始化实验（2026-08-27）

- 实验开关：`EngineConfig.dit_meta_load`，默认 `False`；`run_fsdp_experiment.py` 显式设置为 `True`，因此未改变生产调用方的默认行为。
- 日志：`experiment_logs/fsdp_baseline/101-20260827-141715.log`。
- 方法：在 meta device 上构建 DiT 骨架，通过 `load_state_dict(..., assign=True)`直接挂载 BF16 safetensors 参数，并在 FSDP 前重新物化 checkpoint 未包含的 RoPE `freqs`；强制检查不存在残留 meta parameter/buffer。
- 结果：init-only 成功，`engine_load` 从 276.538 秒降至 84.962 秒，减少 191.576 秒（69.3%，约 3.25 倍加速）；`pipeline_load` 从 273.686 秒降至约 82.4 秒。
- DiT 构建从最多 181.140 秒降至 0.169 秒；checkpoint read/validate/assign 合计约 0.07 秒，包含 freqs 物化和 GC 的聚合阶段最多 0.219 秒；DiT FSDP 包装最多 4.900 秒。
- 当前剩余最大瓶颈为未优化的 T5，两个 rank 分别为 69.881 秒和 70.711 秒。
- 本轮仅验证模型加载、参数 dtype/meta 完整性、FSDP 包装和 readiness 同步，尚未通过实际 DiT forward 或完整视频推理验证数值正确性；进入后续优化前应先做一次固定 seed 的完整推理回归。

## 16. T5 meta+assign 与完整推理回归（2026-08-27）

- 实验开关：`EngineConfig.t5_meta_load`，默认 `False`；实验脚本同时启用 T5 和 DiT meta 加载。
- T5 使用 meta 构建、`torch.load(weights_only=True, mmap=True)` 和 `load_state_dict(assign=True)`，FSDP 前强制检查无残留 meta parameter/buffer。
- init-only 日志：`experiment_logs/fsdp_baseline/101-20260827-142145.log`。结果成功，T5 总阶段从约 70 秒降至最多 2.937 秒，`engine_load` 从最初 276.538 秒降至 16.511 秒，`pipeline_load` 最多 13.881 秒。
- 完整推理日志：`experiment_logs/fsdp_baseline/101-20260827-142221.log`。初始化成功（`engine_load=16.857` 秒），T5 forward 和至少两个 DiT sampling step 成功，证明 meta 参数及重新物化的 RoPE `freqs` 已进入真实计算。
- 完整推理在第 1/4 segment、约 2/6 sampling step 后长时间无进展：GPU 5 持续约 100% SM、GPU 4 等待，未发现新的 Xid/SXid。用户确认此前也发生过类似停顿，因此当前证据不足以将其归因于 meta 优化。
- 本次卡死任务已人工终止；孤立的两个 worker PID 也已清理，GPU 4、5 最终均为 0 MiB。没有生成最终 MP4；这里仅表示 T5+DiT meta 优化后的完整回归未通过，并不表示新命令行入口从未端到端成功。
- 第二次完整回归日志：`experiment_logs/fsdp_baseline/101-20260827-143208.log`。初始化再次成功（`engine_load=16.634` 秒），随后稳定复现相同停顿：第 1/4 segment 的 2/6 sampling step 后无进展，GPU 4 为 0%、GPU 5 为 100%，显存分别约 39267 MiB 和 40217 MiB。任务已通过定向 SIGTERM 清理，无孤立进程、无 MP4，GPU 4、5 最终均回到 0 MiB。
- 两次优化后完整回归在相同位置和相同 GPU 利用率形态停顿，后续不应继续盲目重跑；应在 sampling step/FSDP collective 边界增加 rank-aware 埋点或启用 NCCL flight recorder，以确认两个 rank 的 collective 序列是否发生分歧。由于用户确认优化前也曾出现类似问题，现阶段仍不能仅凭复现位置认定 meta 加载是根因。

## 17. 第三次完整推理与逐步定位（2026-08-27）

- 日志：`experiment_logs/fsdp_baseline/101-20260827-143713.log`；未生成 MP4。
- 为采样循环增加 rank-aware 的 step、conditional forward 和 scheduler step 埋点，并在 forward/step 后显式执行 CUDA synchronize，避免异步 CUDA 提交使完成日志产生歧义。
- 初始化再次成功：两个 rank 的 `engine_load` 均为 16.654 秒，`pipeline_load` 最大为 14.024 秒。
- rank 0/1 均完成第 1、2 个 sampling step；conditional forward 每步约 17.6--17.7 秒，scheduler step 不超过 0.003 秒。两个 rank 随后都进入 segment 1、step 3 的 `sampling_cond_forward`，但都没有输出 complete。
- 受限执行环境内按命令行特征过滤时未匹配到 worker/torchrun，且其中的 `/proc` 视图也不可见对应 PID；这不是宿主机进程已经退出。通过宿主机进程视图确认两个 worker 仍存活，并已成为 PPID 1 的孤儿进程：PID 3534737 是 rank 0/local rank 0，PID 3534738 是 rank 1/local rank 1。最外层 launcher 的 Ctrl+C 没有结束它们。
- 宿主机 `nvidia-smi` 正常：GPU 4 为 0% utilization、39267 MiB，GPU 5 为 100% utilization、40217 MiB。rank 0 主线程主要在 `hrtimer_nanosleep` 等待；rank 1 主线程持续接近 100% CPU，且 GPU 5 保持 100%。这与前两次复现的等待/忙碌形态一致，故障边界为第三次 DiT conditional forward 内部的 GPU/FSDP 执行。
- 本次故障期间内核日志没有出现新的 Xid/SXid。系统历史日志存在 NVSwitch fatal SXid 10003/heartbeat timeout，以及 NVSwitch kernel 575.57.08 与 user 535.129.03 的版本不一致记录，但当前证据不能证明这些历史记录导致了本次停顿。
- 在清理 PID 3534737、3534738 前不要启动新实验。若要继续定位，应先保留现场采集 rank 线程栈/NCCL 状态，或清理后启用 NCCL flight recorder 重新运行，以定位 step 3 内部的 FSDP collective。
- 宿主机现场诊断：`nvidia-smi pmon/dmon` 显示 rank 1/GPU 5 长期为 99--100% SM，但显存利用率、PCIe 流量均为 0，功耗仅约 85 W；GPU 4 为 0% SM、约 63 W。两次读取 NVLink throughput counters 完全相同，证明卡住期间没有 NVLink 数据传输。这不像正常 DiT 计算，更符合低功耗等待/自旋 kernel（包括可能的 NCCL 等待 kernel）。
- NVLink error counters 中，GPU 5 Link 5 累计 replay=66348、CRC=65535（计数饱和）；GPU 4 Link 6 累计 replay=31461，GPU 5 Links 8/9/10/11 也有 replay。短时间复读时计数未继续增加，因此这些是自上次 reset 起的累计异常，属于重要硬件/链路风险信号，但不能单独证明本次卡死由它触发。
- GPU 4、5 当前 ECC、row remap、PCIe replay 和 `GPU Recovery Action` 均无异常，本次期间内核日志也没有新增 Xid/SXid。DCGM host engine 未运行，无法采集 Tensor/DRAM profiler 指标。
- 由于两个 worker 已成为 PPID 1 的孤儿进程，Yama ptrace policy 禁止普通用户 gdb attach；无密码 sudo 也不可用，因此未取得决定性的原生/Python 栈。进程超过约 10 分钟仍没有 NCCL timeout/trace 输出，当前任务又未预先启用 flight recorder，无法从存量进程补取 collective 序号。
- 用户随后要求清理两个孤儿 worker；执行前精确复核时 PID 3534737、3534738 已自行退出，因此 SIGTERM 返回 `No such process`，没有信号发送到其他进程。GPU 4、5 均已释放至 0 MiB、0% utilization，宿主机无本次实验残留进程。日志未补出 timeout/trace 信息。

## 18. 实验 GPU 切换与健康检查（2026-08-27）

- 后续实验从物理 GPU 4、5 切换到物理 GPU 2、3；单卡固定使用 GPU 2，双卡固定使用 GPU 2、3。`run_fsdp_experiment.py` 的硬编码绑定和环境校验已同步修改为 `CUDA_VISIBLE_DEVICES=2,3`。
- 检查时 GPU 2、3 均为空闲状态：0% utilization、0 MiB used。PCI 地址分别为 `0000:47:00.0` 和 `0000:4e:00.0`。
- 两卡各 12 条 NVLink 均为 active/25 GB/s；所有 NVLink replay、recovery、CRC error counters 均为 0。两卡 PCIe replay 均为 0，`GPU Recovery Action=None`。
- 内核日志没有指向 GPU 2/`47:00` 或 GPU 3/`4e:00` 的 NVRM Xid。历史 Xid 62/45/74/154 明确指向 `90:00`，即原 GPU 5。
- GPU 2 的 volatile/aggregate ECC 和 row remap 均为 0。GPU 3 当前 volatile ECC 为 0，生命周期 aggregate DRAM correctable=15、correctable remapped rows=2；无 uncorrectable error、无 pending row、无 remapping failure。这与 GPU 5 的大量 NVLink replay/CRC 错误不是同类或同量级问题，当前可用于实验，但后续每次实验前后应复读 GPU 2、3 的 NVLink/ECC counters。

## 19. GPU 2、3 完整推理对照实验（2026-08-27）

- 日志：`experiment_logs/fsdp_baseline/101-20260827-145329.log`；输出目标为 `experiment_outputs/fsdp_baseline/101-20260827-145329.mp4`，但未生成 MP4。
- 实验前 GPU 2、3 均为 0 MiB/0%，两卡所有 NVLink replay/recovery/CRC counters 均为 0。
- 初始化成功：两个 rank 的 `engine_load` 均为 17.949 秒，`pipeline_load` 最大为 15.229 秒。
- rank 0/1 均完成 segment 1 的 sampling step 1、2，每次 conditional forward 约 17.6--17.8 秒；随后两个 rank 都进入 step 3 的 `sampling_cond_forward` 且不再 complete，与 GPU 4、5 上三次复现的位置完全相同。
- 卡死现场为 local rank 0/物理 GPU 2：0% utilization、39267 MiB；local rank 1/物理 GPU 3：100% utilization、40217 MiB。实验后复读 GPU 2、3 的全部 NVLink error counters 仍为 0。
- 本次卡死任务通过精确 SIGTERM 终止 worker PID 3554391、3554392 和 torchrun PID 3554317；无残留进程，GPU 2、3 均释放到 0 MiB/0%。
- 关键结论：相同卡死位置和“rank 0 等待、rank 1 忙碌”形态在健康且 NVLink error=0 的 GPU 2、3 上复现，因此原 GPU 5 的历史 Xid/NVLink 错误不是复现该故障的必要条件。现有证据明显更支持 rank/软件路径相关的 FSDP collective、CUDA stream 同步或执行序列问题，而非单张物理 GPU 5 的硬件故障。

## 20. 新命令行入口的标准加载成功基线（2026-08-27，补记）

- 在自动保存日志功能加入前，新命令行入口曾于 13:13--13:26 完整成功运行一次；当时没有保留 stdout/stderr 日志，但输出文件仍存在：`experiment_outputs/fsdp_baseline/101-20260827-131318.mp4`，大小 19587416 bytes。
- `ffprobe` 验证该输出为 H.264、512×896、30 fps、297 帧、9.9 秒；当前固定输入 `testdata/101/driving_video.mp4` 同样为 512×896、30 fps、297 帧、约 9.9 秒，因此这是当前测试数据的新入口完整成功基线，而不是 2026-08-25 的旧服务样例。
- 该成功运行发生在 DiT meta 实验（约 14:17）和 T5 meta 实验之前。当时 T5 在 CPU 正常构造并使用标准 `load_state_dict(assign=False)`；DiT 在 CPU 正常构造、转换 BF16 后使用标准 `load_state_dict(assign=False)`，随后进行相同的双卡 FSDP 包装。
- 此后同时启用 T5/DiT meta+assign 的四次完整回归均在 segment 1、sampling step 3 的 DiT conditional forward 内以相同 rank 形态卡住，其中一次已在健康 GPU 2、3 上复现。由此应将 meta+assign 加载优化或其伴随改动列为首要嫌疑；GPU 5 硬件错误和测试数据差异已不能解释该对照。
- 下一步应先在当前代码和 GPU 2、3 上恢复 `t5_meta_load=False, dit_meta_load=False`，复现已知成功基线；成功后逐项启用 T5 meta、DiT meta，以确定具体责任路径。不要再使用 2026-08-25 旧服务日志作为该判断的主要对照。

## 21. 三组加载模式并行对照实验（2026-08-27）

- 为避免使用 GPU 4、5，命令行入口新增 `--physical-gpus`，仅允许从物理 GPU 0、1、2、3、6、7 中选择两个互异编号；本轮并行运行三组双卡作业。实验前六张卡均为空闲，GPU 0、1、2、3、6 的 NVLink error counters 全为 0；GPU 7 只有 Link 9 的历史 replay=1，CRC/recovery 均为 0。
- 标准 T5 + 标准 DiT，GPU 2、3：日志 `experiment_logs/fsdp_baseline/101-20260827-150449.log`。初始化成功，`engine_load=272.008` 秒；进入视频生成后停在首个 segment 的预处理阶段，没有输出 `Processing segment 1/4`，物理 GPU 2 为 100%、GPU 3 为 0%。
- meta T5 + 标准 DiT，GPU 0、1：日志 `experiment_logs/fsdp_baseline/101-t5meta-ditstandard-gpu01-20260827-150800.log`。初始化成功，`engine_load` 最大为 208.160 秒；两个 rank 均完成 sampling step 1、2，随后停在 segment 1、step 3 的 `sampling_cond_forward`。
- 标准 T5 + meta DiT，GPU 6、7：日志 `experiment_logs/fsdp_baseline/101-t5standard-ditmeta-gpu67-20260827-151000.log`。初始化成功，`engine_load=88.302` 秒；两个 rank 均完成 sampling step 1、2，随后同样停在 segment 1、step 3 的 `sampling_cond_forward`。
- 两个进入 step 3 的混合加载作业每步 conditional forward 均约 17.6--17.8 秒；观察超过正常单步时长后没有任何 rank 输出 complete。现场呈现每组一张卡 100%、另一张卡 0% 的相同不对称利用率。
- 三个作业均未生成 MP4。确认无进展后通过各自前台会话发送 Ctrl+C 精确终止，所有 torchrun/worker 均随父进程退出；GPU 0、1、2、3、6、7 最终均为 0 MiB/0%，GPU 4、5 全程未使用。
- 结论：meta 开关不是本轮卡死的必要条件；纯标准加载作业在并发场景下也发生了活性故障。因此不能再把“是否使用 meta+assign”视为唯一变量或直接根因。不过纯标准组没有到达 step 3，三组又并发共享主机资源，本轮不能严格复现单作业条件下的成功基线，也不能据此排除 meta 优化伴随的其他代码改动。下一步应停止并行 A/B，先单独运行纯标准组；若仍失败，则比较 13:13 成功运行之后的源代码改动，并优先定位 forward/FSDP 执行序列。

## 22. 移除采样 CUDA synchronize 后的单作业标准加载实验（2026-08-27）

- 移除了此前为逐步定位新增的三处 `torch.cuda.synchronize(self.device)`：conditional forward、unconditional forward 和 scheduler step 之后；保留分段清理、最终同步和 warmup 等原有生命周期同步点。
- 单独在物理 GPU 2、3 上运行标准 T5 + 标准 DiT，实验前两卡均为空闲，全部 NVLink replay/recovery/CRC counters 为 0。
- 日志：`experiment_logs/fsdp_baseline/101-20260827-152158.log`；初始化成功，两个 rank 的 `engine_load` 均为 268.976 秒。
- 本次顺利越过了上一轮纯标准并发作业偶发停住的分段预处理，并完成 segment 1 的 sampling step 1、2；随后两个 rank 均进入 step 3 的 `sampling_cond_forward`，没有输出 complete，再次复现相同卡死位置。
- 用户确认不再继续等待后，通过前台会话 Ctrl+C 精确终止作业；所有 worker/torchrun 均退出，GPU 2、3 回到 0 MiB/0%，未生成 MP4。
- 结论：新增的三处 device-wide CUDA synchronize 不是该 step 3 卡死的必要条件。后续应继续二分 meta 优化期间对公共标准路径的其他源代码改动，或直接恢复 13:13 成功时的 legacy standard 路径做单作业对照。

## 23. 完整回滚至成功版本后的 legacy-standard 测试（2026-08-27）

- Git 历史显示 13:13 成功运行之后，提交 `148837e`（14:08，`Added some logs`）修改了 `wan/scail.py`、模型、VAE、CLIP 和分布式代码；之后工作区又加入了 meta/T5/DiT 优化与采样诊断。运行时代码现已全部恢复到 `f26e969`（13:20，仅提交 `testdata/101`）中的内容，并通过逐文件 `git diff --exit-code f26e969 -- ...` 验证完全一致。
- 保留了实验日志、输出目录、上下文文档、`.gitignore`，以及外层 CLI 的日志保存/时间戳/GPU 绑定能力；CLI 已移除对 `t5_meta_load`、`dit_meta_load` 配置字段的依赖，不改变 legacy 模型执行路径。
- 第一次尝试日志：`experiment_logs/fsdp_baseline/101-20260827-153131.log`。该作业发生用户此前确认的偶发加载卡死：两个 rank 均进入 `dit_fsdp_wrap` 后不再 complete，物理 GPU 2 为 100%、GPU 3 为 0%。终止并释放两卡后进行一次受控重试。
- 第二次尝试日志：`experiment_logs/fsdp_baseline/101-20260827-153842.log`。初始化成功，两个 rank 的 `engine_load` 均为 271.447 秒；进入 segment 1 后 tqdm 正常完成 1/6、2/6，随后第三步长时间无进展，与后续代码逐步埋点确认的 step 3 卡死位置一致。
- 用户请求的是完整测试，但第二次作业已明确复现卡死，故没有继续等待；通过前台 Ctrl+C 精确终止，所有 worker/torchrun 均退出，GPU 2、3 回到 0 MiB/0%，两次均未生成 MP4。
- 关键结论：13:13 成功之后的源代码修改不是 step 3 卡死的必要条件；完整恢复 legacy 运行时代码仍可复现。因此应把后续排查重点转向成功运行与当前运行之间的非代码状态差异，例如 GPU 对/拓扑、驱动或 NCCL 状态、进程启动环境、资源并发、输入/模型文件缓存状态及其他宿主机运行状态。当前成功样例使用的物理 GPU 对也应复核，但继续遵守不使用 GPU 4、5 的约束。

## 24. 跨机器复现实验（2026-08-27，补记）

- 在 legacy-standard 路径已经完整回滚、但原机器仍复现卡死后，将当前实验工作区迁移到另一台服务器做机器变量对照。用户确认新服务器的硬件和软件配置与原服务器完全相同；实验仍不经过容器，继续使用相同的 Conda 环境、模型目录、固定 `testdata/101` 输入和双卡 FSDP 命令行入口。
- 在新服务器上重新运行了一次完整推理实验，使用的远端日志名为 `experiment_logs/fsdp_baseline/101-20260827-155730.log`。该日志只存在于另一台服务器，切回原服务器时没有同步，因此当前工作区不应将它当作可直接读取的本地证据。
- 新服务器上的实验仍出现了与原服务器相同的推理失败/卡死现象，没有完成端到端输出。由于远端日志未同步，本记录不补写当前无法重新核对的具体 rank、block、显存数值或进程 PID。
- 随后工作环境切回原服务器；另一台服务器上最后写入的上下文修改和实验日志均未带回，也没有必要据此更改当前源代码。
- 该跨机器对照排除了“仅由原服务器的临时 GPU/NVLink/驱动状态触发”这一解释，支持继续从两台机器共有的软件执行路径和显存管理行为排查。后续在原服务器上的诊断最终确认首发故障为 CUDA allocator 碎片化导致的 rank 0 OOM，NCCL 等待是次生现象。

## 25. FSDP 卡死根因与 CUDA allocator 验证（2026-08-27）

- 为 legacy-standard 路径加入了仅由 `--diagnose-fsdp` 启用的诊断能力：各 DiT block 的 rank-aware 前后埋点、CUDA allocated/reserved 显存、NCCL desync/flight-recorder 信息，以及 120 秒 process-group watchdog。诊断开关关闭时不改变模型 forward 路径。
- 根因日志：`experiment_logs/fsdp_baseline/101-20260827-160920.log`。两个 rank 均完整完成 sampling step 1、2；step 3 时 rank 0 完成 block 0 后进入 block 1，rank 1 则继续完成 block 1--3 并进入 block 4，首次精确定位到 rank 执行序列发生分叉的位置。
- rank 0 的真实异常是 `torch.OutOfMemoryError`：在 `wan/modules/model_scail2.py` 的 block 1 FFN/GELU 中申请 1.26 GiB 失败。此时 GPU 仍有 1.15 GiB 物理空闲，PyTorch 已分配 32.23 GiB、另有 5.01 GiB reserved-but-unallocated，符合 CUDA caching allocator 碎片化，而不是模型总显存需求绝对超过 40 GiB。
- rank 0 OOM 后没有加入下一次 FSDP all-gather；rank 1 已继续提交后续 collective，因而表现为一张卡 0%、另一张卡 100%。watchdog 最终报告 rank 0 完成 collective #302 但未加入 #303，rank 1 已提交到 #304；`SeqNum=303` 的 `_ALLGATHER_BASE` 在 120 秒后超时。此前观察到的 NCCL“卡死”是 rank 0 OOM 后的次生通信等待，不是首发故障。
- 修复验证日志：`experiment_logs/fsdp_baseline/101-20260827-161806.log`。命令行额外启用 `--expandable-segments`，即在 torch worker 启动前设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；其余仍为 legacy-standard 双卡 FSDP 路径和同一组 `testdata/101` 输入。
- 启用 expandable segments 后，长 segment 的初始 reserved 显存由失败实验约 38.0--38.2 GiB 降至约 29.3 GiB，运行高峰约 38.2 GiB；两个 rank 的显存轨迹保持一致。四个 segment、每段 6 个 sampling step 均完成，没有 OOM、rank 分叉或 watchdog timeout，进程以 exit code 0 正常退出。
- 输出 `experiment_outputs/fsdp_baseline/101-20260827-161806.mp4` 已由 `ffprobe` 验证：H.264、512x896、30 fps、297 帧、9.9 秒、19587416 bytes。结束后 GPU 2、3 均为 0 MiB/0%，无残留作业。
- 结论：当前反复出现的 step 3 卡死已经有确定的软件根因和可复现实验修复；它与 meta/standard 加载开关、GPU 5 历史硬件错误及机器迁移均无必然关系。后续实验至少应启用 expandable segments，并保留 watchdog 作为故障快速暴露手段。日志中的 IB/RoCE HCA 合并 warning 在本次成功运行中仍出现，因此不是本轮卡死的必要条件，但后续可独立清理 NCCL HCA 配置。

## 26. 恢复全部启动加载优化并完成 init-only 回归（2026-08-27）

- 在确认首发故障是 CUDA allocator 碎片化、而不是 meta 加载后，从本机 Git 不可达对象中找回了此前实际运行过的优化代码快照，并将加载相关改动重新合并到当前 legacy-standard+诊断代码上；没有用手工推测替代旧实现，也没有覆盖当前 watchdog、FSDP block 诊断或 expandable allocator 开关。
- `EngineConfig` 重新加入 `t5_meta_load` 和 `dit_meta_load`，默认仍为 `False`，因此没有改变其他调用方的默认加载方式；实验命令行入口固定同时启用两项优化，并在日志头标记 `T5=meta-assign, DiT=meta-assign`。
- T5 恢复为 meta device 构建，使用 `torch.load(weights_only=True, mmap=True)` 读取 checkpoint，再通过 `load_state_dict(strict=True, assign=True)` 直接绑定参数；FSDP 前检查不存在残留 meta parameter/buffer。VAE 原路径本来已经是 meta+assign，无需重复修改。
- DiT 恢复为 meta device 构建和 safetensors `assign=True` 加载；checkpoint 未包含且不注册为 parameter/buffer 的 RoPE `freqs` 在 CPU 上重新物化，并在 FSDP 前检查所有 parameter、buffer 和 `freqs` 都已离开 meta device。
- 第一次 init-only 启动日志为 `experiment_logs/fsdp_baseline/101-20260827-163914.log`；该次在受限执行沙箱内因本地 TCPStore 无法解析/连接 `localhost` 而在 process group 建立前失败，未进入模型加载、未占用 GPU。随后以完全相同的程序参数在允许本机 rank 通信的宿主执行环境重跑。
- 有效验证日志：`experiment_logs/fsdp_baseline/101-20260827-164018.log`。物理 GPU 2、3，命令行启用 `--init-only --expandable-segments`；两个 rank 均成功完成 T5/DiT meta 构建、checkpoint assign、FSDP 包装、barrier 和 ready，进程 exit code 0。
- 本轮两个 rank 的 `engine_load` 均为 17.828 秒，`pipeline_load` 最大 15.227 秒；T5 总阶段最大 3.162 秒，T5 构建约 0.096 秒、mmap read 约 0.030 秒、assign 约 0.007 秒；DiT 构建最大 0.163 秒、checkpoint copy 聚合阶段最大 0.234 秒、FSDP 包装最大 5.017 秒。性能与此前 meta 优化实验一致。
- init-only 按设计没有进入推理、没有生成 MP4。结束后 GPU 2、3 均回到 0 MiB/0%，无残留作业。退出时仍有 PyTorch FSDP weak-reference `PyInterpreter.cpp` warning，但不影响 readiness 或 exit code；完整推理组合回归尚未在本次修改后重新执行，后续运行必须继续启用 expandable segments。

## 27. 恢复初始化日志与 AMP warning 修复（2026-08-27）

- 恢复提交 `148837e` 中与启动相关的分段计时日志：DiT checkpoint header、T5、VAE、CLIP、DiT model construct/checkpoint/FSDP wrap，以及 engine/process-group/barrier/ready；同时保留 meta 加载新增的 T5 construct/read/assign 和 DiT read/validate/assign 子阶段。没有恢复曾用于卡死定位、会在每个 sampling forward 后强制 `torch.cuda.synchronize()` 的临时日志代码。
- 将 `wan/scail.py`、模型、VAE、CLIP 和分布式路径中的旧 `torch.cuda.amp.autocast` / `amp.autocast` 全部恢复为 `torch.amp.autocast("cuda", ...)`，并删除不再使用的 `torch.cuda.amp` import。源码扫描已确认 `wan/**/*.py` 中没有残留 `torch.cuda.amp`。
- 回归日志：`experiment_logs/fsdp_baseline/101-20260827-164502.log`。物理 GPU 2、3，`--init-only --expandable-segments`，T5/DiT 均使用 meta-assign；两个 rank 的 `engine_load` 均为 17.821 秒，完整输出初始化阶段日志并达到 ready，进程 exit code 0。
- 回归日志中没有 `FutureWarning`、`torch.cuda.amp`、OOM 或 traceback；结束后 GPU 2、3 均为 0 MiB/0%。仍可见的 `PyInterpreter.cpp` weak-reference warning 来自 FSDP 对象退出清理，与本次已消除的 AMP deprecation warning 不同。

## 28. 加速加载后的完整推理回归（2026-08-27）

- 在物理 GPU 2、3 上运行完整双卡 FSDP 推理，固定使用 `testdata/101`，T5 和 DiT 均启用 meta-assign 加速加载，同时启用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；本轮未启用高噪声的逐 block FSDP 诊断。
- 执行命令：`python -u run_fsdp_experiment.py --physical-gpus 2,3 --expandable-segments`。日志：`experiment_logs/fsdp_baseline/101-20260827-164739.log`；输出：`experiment_outputs/fsdp_baseline/101-20260827-164739.mp4`。
- 两个 rank 的 `engine_load` 均为 17.862 秒，`pipeline_load` 最大 15.112 秒；T5 meta-assign 总阶段最大 3.122 秒，DiT meta 构建最大 0.166 秒、checkpoint assign 0.032 秒、FSDP wrap 最大 5.051 秒。两端均完成 ready barrier 后进入生成。
- 四个 segment 全部完成，每段 6 个 sampling step；前三个 81 帧 segment 每步约 17.4 秒，最后一个 57 帧 segment 每步约 10.8 秒。音频合并耗时 2.428 秒，任务 JSON 返回 `status=success`，进程 exit code 0。
- `ffprobe` 验证输出包含 H.264 视频和 AAC 音频：512x896、30 fps、297 帧、9.9 秒、19587416 bytes。输出规格和此前 legacy-standard+expandable 成功基线完全一致。
- 日志中没有 AMP `FutureWarning`、OOM、traceback、NCCL error 或 timeout。退出清理阶段仍有已知的 FSDP `PyInterpreter.cpp` weak-reference warning，但不影响结果保存和正常退出。
- 结束后包括 GPU 2、3 在内的全部 GPU 均为 0 MiB、0% utilization，无实验残留进程。该回归确认 T5/DiT meta-assign 加载优化与 expandable allocator 可以共同完成端到端推理，加载优化本身不会导致此前的 step 3 停顿。
