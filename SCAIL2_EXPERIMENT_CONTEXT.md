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

## 29. T5/CLIP 独立预处理与主流程摘除实验（2026-08-27）

- 新增 `prepare_conditioning.py`，在单张物理 GPU 2 上顺序执行 T5 prompt encoding 和 CLIP reference-image encoding，不加载 VAE 或 DiT。输出使用版本化 safetensors 缓存；当前固定样例为 `experiment_cache/conditioning/101.safetensors`，大小 1420832 bytes。
- 缓存严格绑定 prompt、negative prompt、参考图 SHA-256、目标尺寸，以及 T5/CLIP checkpoint 的路径、大小和 mtime；主推理加载时校验全部元数据和 tensor 契约。三个 tensor 分别为：`text_context=[92,4096]` BF16、`negative_context=[1,4096]` BF16、`clip_context=[1,257,1280]` FP16。
- 预处理日志：`experiment_logs/conditioning/101-20260827-172426.log`。T5 阶段 3.598 秒、GPU peak allocated 11088.6 MiB；CLIP 阶段 2.170 秒、GPU peak allocated 2491.2 MiB；包含模型加载、两次编码、顺序释放和缓存保存的进程总耗时 12.5 秒。
- `EngineConfig.precomputed_conditioning=True` 时要求 `t5_fsdp=False`；Pipeline 构造阶段完全跳过 T5 和 CLIP，generate 阶段只接收已验证的三组 CPU tensor 并移动到当前 rank 的 GPU。未提供缓存、缓存身份不匹配或 tensor dtype/shape 不匹配都会直接失败。
- init-only 日志：`experiment_logs/fsdp_baseline/101-20260827-172514.log`。两个 rank 均跳过 T5/CLIP 并成功完成 VAE、DiT meta-assign、FSDP wrap 和 ready；`engine_load=13.537` 秒，相比包含 T5/CLIP 的 17.862 秒减少 4.325 秒（24.2%）。
- 完整推理日志：`experiment_logs/fsdp_baseline/101-20260827-172549.log`；输出：`experiment_outputs/fsdp_baseline/101-20260827-172549.mp4`。`engine_load=13.771` 秒，四个 segment、每段 6 步全部完成，进程 exit code 0，无 OOM、NCCL timeout、traceback 或 AMP FutureWarning。
- 缓存主流程的 ready-to-finished 推理请求耗时为 436.904 秒；原在线 T5/CLIP 流程为 437.707 秒，减少 0.803 秒（0.18%），扩散采样单步速度不变。预处理的主要收益不是单次 DiT 计算加速，而是让主 worker 不再加载/常驻 T5 和 CLIP，并允许缓存跨任务复用。
- 新旧两次完整输出的 SHA-256 完全相同：`7039c5f231eb64b544c4aa288ea5107411c9e7f51bdcf4c93d125d6e1610680a`。这比仅比较媒体规格更强，确认当前固定 seed、prompt、参考图和输入下，独立预处理没有引入任何输出字节差异。
- 若把 12.5 秒预处理成本计入只使用一次的任务，总时间不会更快；该结构的价值来自离线/异步预处理、相同 prompt/reference 的复用，以及为单卡主 worker 释放模型显存。`experiment_cache/` 已加入 `.gitignore`。

## 30. DiT 长序列激活显存优化（2026-08-28）

- 新增 `--memory-probe`，只执行首个 81 帧 segment 的第一个 diffusion step，跳过 VAE decode/视频写出；诊断模式记录 block 0 内 self-attention、RoPE、cross-attention 和 FFN 的 allocated/reserved/phase peak，并记录最终 latent SHA-256。GPU 2、3 实验前均为 0 MiB、volatile corrected/uncorrected ECC=0。
- 缓存 conditioning 后的原始诊断日志为 `experiment_logs/fsdp_baseline/101-20260828-130305.log`。初始 block 0 最高峰为 self-attention 的 26350.1 MiB；RoPE Q/K 单阶段增量为 3746.8 MiB；完整 FFN 的最高绝对值为 24300.5 MiB。基线 latent SHA-256 为 `f6845ee32b24e01bb80b8f6dfa3467c62119bb3014ef94f65718a40bd8085261`。
- FFN 按 8192 token 分块，同时分块执行 norm/modulation，并预分配 BF16 输出。日志 `101-20260828-130414.log`：FFN 计算峰值从 24300.5 MiB 降至 22515.2 MiB，减少 1785.3 MiB；首步约 17.82 秒，latent SHA-256 与基线完全相同。
- RoPE 保留 FP64/complex128 内部计算，但直接返回输入 BF16 dtype；FlashAttention 原本也会立即将 Q/K 转为 BF16，因此避免了冗余 FP32 Q/K 和 FP32 attention 返回值。日志 `101-20260828-130545.log`：block 峰值降至 25419.9 MiB，减少 930.2 MiB；reserved 约减少 1.72 GiB；latent SHA-256 仍完全相同。
- 将 RoPE 内部改为 FP32/complex64 的日志为 `101-20260828-130707.log`，峰值可进一步降至约 23970 MiB、首步约 17.32 秒，但 latent SHA-256 改变，因此已回退，当前继续使用 FP64/complex128。
- cross-attention 的 image/text 输出合并以及 FP32 block residual 改为安全的原地累加。日志 `101-20260828-130841.log`：cross merge 临时峰值减少 476.9 MiB，cross residual 峰值减少约 477 MiB，后续 allocated 降低约 954 MiB；latent SHA-256 仍完全相同。全局峰值仍由 RoPE 决定。
- BF16 block residual 的可选实验日志为 `101-20260828-131010.log`。它将 block 间 hidden 从约 953.8 MiB 降至 476.9 MiB，但没有降低由 RoPE 决定的全局峰值，并改变 latent SHA-256，因此默认关闭，仅保留 `--bf16-residual` 实验开关。
- 无损组合完整回归命令：`python run_fsdp_experiment.py --conditioning-cache experiment_cache/conditioning/101.safetensors --physical-gpus 2,3 --ffn-chunk-size 8192 --expandable-segments`。日志 `experiment_logs/fsdp_baseline/101-20260828-131130.log`，输出 `experiment_outputs/fsdp_baseline/101-20260828-131130.mp4`；297 帧、30 fps、四个 segment 全部成功，未出现 OOM/NCCL/traceback。
- 优化后 ready-to-finished 为 432.281 秒，缓存 conditioning 基线为 436.904 秒，减少 4.624 秒（1.06%）。新旧 MP4 大小均为 19587416 bytes，SHA-256 均为 `7039c5f231eb64b544c4aa288ea5107411c9e7f51bdcf4c93d125d6e1610680a`，确认完整输出逐字节一致。
- 在提交 `a5090cb` 后新增可选 `--rope-chunk-size`，继续保留 FP64/complex128 RoPE，只按 token 分块转换和旋转。8192-token 诊断日志为 `101-20260828-142607.log`：Q/K RoPE 峰值分别从 24943.1/25419.9 MiB 降至 22716.6/23193.4 MiB，block 0 最高峰降至 23973.1 MiB，较未分块无损组合减少 1446.8 MiB；reserved 从 25740 降至 24700 MiB。首步 latent SHA-256 仍为 `f6845ee32b24e01bb80b8f6dfa3467c62119bb3014ef94f65718a40bd8085261`。
- RoPE 分块完整回归日志为 `101-20260828-142743.log`，输出为 `101-20260828-142743.mp4`。297 帧、19587416 bytes，SHA-256 仍为 `7039c5f231eb64b544c4aa288ea5107411c9e7f51bdcf4c93d125d6e1610680a`，与未分块完整输出逐字节一致。Ready-to-finished 为 426.297 秒，比未分块无损组合少 5.984 秒；这是单次运行结果。分块后最高点转移到 FlashAttention 23973.1 MiB，Cross Q/K/V 23965.9 MiB 紧随其后。
- FlashAttention wrapper 对完整长度 K/V 增加 `flatten` view fast path，只有变长或含 padding 时才执行切片和 `cat`；原始 Q/K 的释放时机暂未修改。诊断日志 `101-20260828-150011.log`：Self FlashAttention 峰值从 23973.1 降至 23019.4 MiB，减少 953.7 MiB，阶段结束 allocated 仍为 23011.9 MiB；block 最高点转移到 Cross Q/K/V 的 23965.9 MiB。首步 latent SHA-256 保持不变。
- K/V fast path 完整回归日志为 `101-20260828-150127.log`，输出 `101-20260828-150127.mp4`；297 帧、19587416 bytes，SHA-256 仍为 `7039c5f231eb64b544c4aa288ea5107411c9e7f51bdcf4c93d125d6e1610680a`。Ready-to-finished 为 424.583 秒，比 fast path 前少 1.713 秒；这是单次运行结果。
- Cross-attention query 生命周期优化不做分块：调用方不再预先生成并保活 `self.norm3(x)`，而是把原始 hidden 和 query norm 传入 cross-attention；内部仍以完整长度执行相同的 LayerNorm、Q Linear 和 RMSNorm，并在 Q projection 返回后立即释放 953.8 MiB 的 FP32 归一化输入。诊断日志 `101-20260828-152305.log`：Cross Q/K/V 峰值从 23965.9 降至 23012.1 MiB，阶段结束 allocated 从 22073.2 降至 21119.5 MiB；block 0 最高峰转移到 K RoPE 的 23193.4 MiB，较修改前降低 772.5 MiB，两个 rank 最大 reserved 为 23740 MiB。首步 latent SHA-256 保持不变。
- Cross query 提前释放的完整回归日志为 `101-20260828-152421.log`，输出 `101-20260828-152421.mp4`。任务正常完成四个 segment、297 帧，ready-to-finished 为 424.714 秒。新旧输出大小均为 19587416 bytes，SHA-256 均为 `7039c5f231eb64b544c4aa288ea5107411c9e7f51bdcf4c93d125d6e1610680a`，确认最终 MP4 逐字节一致。

## 31. 完整推理设备级显存剖析（2026-08-28）

- 新增 `--full-memory-profile`，与会截断推理的 `--memory-probe` 互斥。该模式完整执行 4 个 segment、24 个 diffusion step、VAE decode 和输出；记录两个 rank 的初始化/参考图 encode/segment prepare/每个 DiT block/scheduler/cleanup/VAE decode/history encode，并由父进程每 200 ms 采集物理 GPU NVML `memory.used`。
- 完整日志为 `experiment_logs/fsdp_baseline/101-20260828-155713.log`，共 6759 行；输出为 `experiment_outputs/fsdp_baseline/101-20260828-155713.mp4`。任务 exit code 0，297 帧、19587416 bytes，SHA-256 仍为 `7039c5f231eb64b544c4aa288ea5107411c9e7f51bdcf4c93d125d6e1610680a`。
- segment 1 / step 1 的 block 0 峰值仍为 23193.4 MiB；block 1--39 因 FP32 residual 多保留 476.9 MiB，峰值为 23670.3 MiB。完整 24 step 中最高 DiT allocated 为 23754.3 MiB，出现在 segment 2/3 的 step 6；长 segment 的最高 reserved 为 26342 MiB，同步 device used/NVML 为 27467 MiB。
- 约 27G 的 DiT 设备占用可拆为约 23734.1 MiB live allocated、2607.9 MiB allocator cached/unallocated 和 1124.8 MiB CUDA context/NCCL/库/驱动占用。此前 block 0 报告与外部监控不一致的主要原因由此确认：统计范围和指标口径均不同。
- 全任务最高点不是 DiT，而是仅由 rank 0/物理 GPU 2 执行的 segment 1 VAE decode：peak allocated=25617.3 MiB、reserved=28600 MiB、同步 device used=29724.8 MiB，200 ms NVML 峰值=29725 MiB。物理 GPU 3 不执行 decode，全任务 NVML 峰值保持为 DiT 阶段的 27467 MiB。
- 详细分阶段数据、每个 segment/step 的 DiT 峰值和 VAE 分解见 `SCAIL2_FULL_MEMORY_PROFILE_REPORT.md`。若后续目标是降低 DiT live tensor，应继续处理后续 block FP32 residual 和多个约 23 GiB 的 attention 峰值；若目标是降低整条任务物理显存峰值，则 VAE decode 与常驻 DiT shard 的重叠优先级更高。

## 32. 单 segment VAE decode 逐操作显存剖析（2026-08-28）

- 新增 `--vae-memory-profile`：只执行第一个 81 帧 segment，但保留完整 6 步 diffusion、21 个 latent 时间步的 VAE decode 和短视频输出；父进程继续以 200 ms 采集 NVML。该模式绕过正式 297 帧输出的发布/音频校验，专用于实验。
- 首次日志 `101-20260828-172328.log` 已完整采集 VAE 数据，但单 segment 仍触发 297 帧校验，导致 rank 0 异常后与 rank 1 barrier 不对称等待；手动终止后无残留进程。随后修正实验模式结束路径，并把逐操作埋点限制为 decode 阶段，避免干扰 reference/pose encode。
- 有效日志为 `experiment_logs/fsdp_baseline/101-20260828-173206.log`；输出 `experiment_outputs/fsdp_baseline/101-20260828-173206.mp4` 已验证为 H.264、512x896、30 fps、81 帧，SHA-256=`8af1377b5c162c70502bbfd9657c5723c3313cd32e34df9f509e90f5d0471a6b`。进程正常退出，GPU 2、3 回到 0 MiB/0%，volatile uncorrected ECC=0。
- decode 开始 allocated=16654.5 MiB；最后时间步的 32 个 causal cache 合计 4137.875 MiB，累计 77 帧输出为 404.250 MiB。稳定 cache 按分辨率为 112x64: 231.875、224x128: 546、448x256: 1008、896x512: 2352 MiB。
- 896x512 residual causal conv 是 live allocated 最高点：当前 allocated=24581.2 MiB，operation peak=25596.3 MiB。时间 cat 后 allocated=23903.0；pad 返回后为 23909.2、pad operation peak=24917.2，精确确认 1008 MiB cat 只在 pad 执行时与 1014.196 MiB padded tensor 重叠，返回后已释放。
- 最终 causal conv 的 padded input 1014.196 MiB、output 672 MiB 和约 1015.1 MiB cuDNN workspace 形成最高峰。最终输出累积 `cat` 的 operation peak 仅 21661.2 MiB，不是主要矛盾。clear cache 后 allocated 从 21236.0 降至 17098.1 MiB，与释放 4137.875 MiB cache 完全吻合。
- 本轮 NVML 峰值为物理 GPU 2 的 29785 MiB；不 decode 的 GPU 3 为 27467 MiB。详细阶段表、cache 分布和 allocator/device 口径解释已补充到 `SCAIL2_FULL_MEMORY_PROFILE_REPORT.md`。
- 另新增面向 buffer 生命周期的独立报告 `SCAIL2_VAE_MEMORY_PROFILE_REPORT.md`，逐项列出 latent、各分辨率 feature、32 个 causal cache、temporal cat、padded input、卷积输出/workspace、空间上采样和 21 步输出累计；该报告与完整推理总览分开，避免将设备峰值报告与 VAE buffer 报告混为一体。
- 按后续决定暂不修改 VAE；数据采集完成后已移除 `--vae-memory-profile`、单 segment 特殊路径、NVML/profile 环境开关和 `wan/modules/vae.py` 中全部同步埋点。`run_fsdp_experiment.py`、`wan/scail.py`、`wan/modules/vae.py` 已与 profiling 前的 Git 版本完全一致；原始日志和报告作为实验记录保留。

## 33. 单卡全量 DiT 常驻可行性（2026-08-28）

- 单卡阶段采用独立入口 `run_single_gpu_experiment.py`，不启动 `torchrun`、不初始化 process group、不开启 FSDP；默认只使用物理 GPU 2，并继续禁止 GPU 4、5。主流程读取预计算 conditioning，因此不加载 T5/CLIP。DiT 使用 meta-assign 从 CPU checkpoint 建模，随后完整 BF16 权重常驻 CUDA；原始 1307 个 CPU checkpoint tensor 保留为后续阶段切换所需的 host master。
- `init-only` 日志为 `experiment_logs/single_gpu/101-20260828-180700.log`。CPU master 和 CUDA DiT 各为 32791088768 bytes（31272.0 MiB）；进程 RSS=31957.4 MiB，ready 时 CUDA allocated=31756.1 MiB、reserved=31768.0 MiB、device used=32266.8 MiB。`engine_load=12.138` 秒，初始化正常退出，证明两份权重分别稳定驻留 CPU/GPU，且没有遗留 FSDP 路径。
- 首步日志为 `experiment_logs/single_gpu/101-20260828-180805.log`。第一个 81 帧 segment 的第一个 diffusion step 完整通过全部 40 个 DiT block，不进行 VAE decode。block 0 最高可见 allocated 为 K RoPE 的 37592.7 MiB，最高 reserved=38494.0 MiB；NVML 采样峰值=39973 MiB。结束后 CUDA model 与 CPU master 均仍完整存在。
- 单卡首步 latent 为 mean=0.001903103、std=0.961709201、SHA-256=`26517d3bf0dd1313f0302ddc2e77ac2e71dd8613c9ca5f5384057faae9764eb3`。双卡 FSDP 对照为 mean=0.001893205、std=0.961710393、SHA-256=`f6845ee32b24e01bb80b8f6dfa3467c62119bb3014ef94f65718a40bd8085261`。字节级哈希不同，但 mean/std 只相差约 9.90e-6/1.19e-6；FSDP flatten/all-gather 与普通参数布局可能选择不同矩阵 kernel 或累加顺序，因此当前只将它记录为跨执行模式的微小数值差异，不能据此判定结果错误。
- 为避免将 VAE offload 混入 DiT 可行性验证，单卡入口增加 `--dit-segment-probe`：只运行第一个 81 帧 segment 的全部 6 个 diffusion step，然后在 VAE decode 和输出编码前退出。底层诊断参数仍默认 1 步，所以既有双卡 `--memory-probe` 行为不变。
- 完整 6 步日志为 `experiment_logs/single_gpu/101-20260828-181018.log`。6/6 步全部成功，扩散循环耗时 100.2 秒，最终 latent mean=0.034449518、std=0.849440634、SHA-256=`eeb8e66a637cc73f93fd5fb271d6eafac0efafa511741e60ea0f862be04dd7e2`。全程没有 DiT CPU offload、OOM、NCCL 或 traceback；NVML 峰值=40073/40960 MiB，只剩 887 MiB（约 2.2%）物理余量。
- 实验结束后 GPU 2 回到 0 MiB、0% utilization，volatile uncorrectable ECC=0。结论是当前 81 帧、FFN/RoPE 8192 分块、expandable segments 配置下，全量 DiT 单卡运行在功能上可行，但显存边际非常薄；在引入 CLIP、VAE decode 或其他 GPU buffer 前必须先做明确的阶段释放/offload，不能与完整 CUDA DiT 直接叠加。
- checkpoint 逐 block 统计表明 40 个 DiT block 均为 770.261 MiB。单卡入口新增 VAE 阶段切换实验：diffusion 完成后将末尾 `blocks.33-39` 的 224 个 CUDA parameter 替换为既有 CPU master 的同 storage 参数，不执行 GPU->CPU 权重拷贝；VAE decode（以及完整流程中的 history encode）结束后再从 CPU master 上传恢复。该功能默认 block 数为 0，FSDP 路径不启用。
- 首次核心实验日志为 `experiment_logs/single_gpu/101-20260828-182559.log`：offload、VAE decode 和 reload 均完成，最后仅因新输出目录不存在而在 MP4 编码时报错；补齐父目录创建后的有效成功日志为 `experiment_logs/single_gpu/101-20260828-182908.log`，进程 exit code 0。
- 7 个 block 实际释放 5391.8 MiB，与 checkpoint 理论值完全一致：segment cleanup 后 allocated=31828.9 MiB，offload 后为 26437.1 MiB；offload 耗时 0.150 秒。VAE decode 开始时 reserved=26510.0 MiB、device used=27028.8 MiB；decode 的 peak allocated=35399.9 MiB、peak reserved=37952.0 MiB、device/NVML=38470.8 MiB，距离 40960 MiB 上限仍有 2489.2 MiB。VAE decode 用时约 6.788 秒。
- reload 7 个 block 耗时 1.511 秒（首次冷态实验为 3.664 秒），恢复后 allocated=31819.7 MiB、device used=32368.8 MiB；residency 校验确认 CUDA model 重新达到完整 31272.0 MiB，CPU master 仍为 31272.0 MiB。全任务 NVML 峰值仍由 DiT 决定，为 40033 MiB，而不是 VAE。
- 单 segment 输出 `experiment_outputs/single_gpu/101-20260828-182908.mp4` 已验证：H.264、512x896、30 fps、81 帧、2.7 秒、5658847 bytes，SHA-256=`7171ab492c7f13ccc5fe74a80661b38243a9cb79d6b38b5328eb367f26e60386`。结束后 GPU 2 为 0 MiB/0%，volatile uncorrectable ECC=0。该实验验证了单次 DiT->VAE->DiT 权重阶段切换；跨 segment 的 history encode 和重复切换仍需在完整单卡推理中验证。
- 完整单卡回归日志为 `experiment_logs/single_gpu/101-20260828-183620.log`，使用物理 GPU 2、预计算 conditioning、完整 CUDA DiT、CPU master 和 7-block VAE 阶段切换。4 个 segment、24 个 diffusion step、4 次 VAE decode、前三段 history encode 和 4 次 DiT reload 全部成功；第二至第四段均在 reload 后继续完成 DiT，确认重复参数替换没有破坏模型。
- 四次 offload 分别耗时 0.136/0.137/0.137/0.141 秒，合计 0.551 秒；四次 reload 为 1.337/1.362/1.359/0.749 秒，合计 4.807 秒；总阶段切换开销 5.358 秒。三个 81 帧 segment 的 DiT 平均均为 16.70 秒/步，57 帧 segment 为 10.31 秒/步；对应双卡 FSDP 为约 16.91--16.99 和 10.49--10.53 秒/步。
- 单卡请求 `started_at` 到 `finished_at` 为 425.044 秒；双卡 FSDP 对照 `101-20260828-152421.log` 为 424.714 秒，单卡只多 0.330 秒（0.078%）。双卡结果额外包含 2.431 秒音频 mux，而单卡配置为无音频；若比较第一个 segment 开始到视频编码前的生成主流程，单卡为 406.104 秒、双卡为 402.992 秒，单卡慢 3.112 秒（0.77%）。单卡较快的 DiT 计算基本抵消了 5.358 秒切换成本。
- 完整输出 `experiment_outputs/single_gpu/101-20260828-183620.mp4` 已由引擎和 ffprobe 验证：H.264、512x896、30 fps、297 帧、9.9 秒、19406099 bytes，SHA-256=`fa56145b030db5f6659be2449ff68fe67850344f52801c97762a677c19c71e70`。与双卡 FSDP 视频逐帧比较为 average PSNR=30.640 dB、SSIM=0.935252；两者并非字节一致，这与先前已记录的 FSDP/非 FSDP 首步微小数值差异一致。
- 完整任务 NVML 峰值为 40075/40960 MiB，仍发生在 DiT；独立 VAE probe 已确认 7-block offload 后 VAE 峰值为 38470.8 MiB。任务结束时 CPU master 和完整 CUDA model 均为 31272.0 MiB，进程正常退出后 GPU 2 回到 0 MiB/0%，volatile uncorrectable ECC=0。按单请求几乎相同的延迟计算，两个独立单卡 worker 可用与一个双卡 FSDP worker 相同的两张 GPU 同时处理两个请求，单位 GPU 吞吐接近翻倍。
- 上述 `183620` 完整单卡实验为了隔离模型阶段曾使用 `output_audio_mode=none`，因此其整个 MP4 哈希不能直接与含 AAC 音频的双卡输出比较。后续已将正式 `--full-inference` 模式恢复为 `output_audio_mode=driving`；init/memory/单 segment probe 仍保持无音频。dry-run 已确认单卡 full inference 和双卡正式路径均为 driving audio。

## 34. 单卡在线 CLIP 编码与阶段 offload（2026-08-28）

- 单卡路径新增 `online_clip_conditioning`：T5 正向/负向 context 继续读取 conditioning cache，CLIP visual encoder 改为 CPU 常驻；每个任务完成 reference VAE 处理后，CLIP 短暂上 GPU 生成 visual context，随后立即回 CPU并执行 CUDA cache 清理。双卡和生产配置默认关闭该开关，现有路径不变。
- 当前仍读取原 v1 conditioning artifact 中的缓存 `clip_context`，但它只用于逐元素校验，不作为本次 DiT 输入。这样无需先迁移缓存 schema，即可确认在线 CLIP 与离线预处理结果是否一致。
- 首步日志为 `experiment_logs/single_gpu/101-20260828-191658.log`。冷态 CLIP CPU checkpoint 加载 8.070 秒，engine load 为 19.849 秒；ready 设备占用仍为 32266.8 MiB，证明 CLIP 初始化没有留下 CUDA 权重。在线编码/offload 耗时 2.721 秒，combined allocated 峰值为 33024.8 MiB；offload 后 allocated=31811.7 MiB、device used=32352.8 MiB。
- 在线和缓存 CLIP context 均为 `[1,257,1280]` FP16，逐元素 `torch.equal=True`、最大绝对差为 0。首步 latent SHA-256 仍为 `26517d3bf0dd1313f0302ddc2e77ac2e71dd8613c9ca5f5384057faae9764eb3`，与此前纯缓存单卡首步完全一致。任务 exit code 0，NVML 峰值为 40061/40960 MiB，仍发生在 DiT。
- 正式完整回归日志为 `experiment_logs/single_gpu/101-20260828-191836.log`，输出为 `experiment_outputs/single_gpu/101-20260828-191836.mp4`。热态 CLIP CPU 加载 4.551 秒，engine load 16.379 秒；在线编码/offload 1.331 秒，阶段 NVML 峰值 34555 MiB。4 个 segment、24 个 diffusion step、4 次 VAE decode、3 次 history encode 和 4 次 7-block offload/reload 全部成功。
- 完整请求 started-to-finished 为 429.555 秒，其中 driving audio mux 为 2.599 秒；此前无在线 CLIP且无音频的单卡请求为 425.044 秒。与含音频的双卡 FSDP 对照 424.714 秒相比，新单卡路径慢 4.841 秒（1.14%），仍只使用一张 GPU。
- 新输出为 H.264 512x896、30 fps、297 帧、9.9 秒，并包含 AAC 9.9 秒音频；文件大小 19652849 bytes，SHA-256=`fcb0871b57305440b8cd33ab8e3960a7d8f81f68c401190addda80ac59137d7e`。视频 elementary stream SHA-256 与旧单卡输出均为 `d873574d3209902804225f21d427b987aab383c80372ca681641ff5adf4fc7a9`；音频 stream SHA-256 与双卡 driving audio 均为 `dce6c3e3d3db3750c5aebabb8cdfc346bcfa050fbeffde966dfb699629236340`。
- 完整任务 NVML 峰值为 40081/40960 MiB，仍由 DiT 决定，剩余约 879 MiB。任务结束后 GPU 正常释放。结论是 CLIP 可以安全回到单卡主流程，但必须只在 reference 阶段短暂驻留 GPU；它增加约 1.3 秒热态请求开销和约 1.2 GiB CPU RSS，不改变视频结果或后续 DiT/VAE 阶段行为。

## 35. 单卡与双卡首个数值差异定位（2026-08-31）

- 对比实验不加载或执行 T5/CLIP，单卡和双卡都读取同一个 `experiment_cache/conditioning/101.safetensors`；其中 visual context 仍是模型必需输入，但来自预计算缓存。两边固定 seed=42，并使用相同的 FFN/RoPE 8192 分块，只执行 segment 1 的第一个 diffusion step。
- 首轮单卡 trace 为 `experiment_logs/single_gpu/101-20260831-112917.log`，双卡 trace 为 `experiment_logs/fsdp_baseline/101-20260831-113234.log`。对模型入口、embedding、block 0 的 modulation/self-attention/cross-attention/FFN 和最终 head 共记录 63 个 tensor 的 shape、dtype 与 SHA-256。
- 最早的输入差异来自 FSDP `MixedPrecision` 的默认 `cast_root_forward_inputs=True`：双卡根 FSDP wrapper 在 forward 前将所有浮点输入转为参数 dtype BF16；普通单卡模型没有这层包装，因此缓存的 `clip_context` 保持 FP16，其他若干输入保持 FP32。大部分 FP32 输入在后续 autocast 算子中仍产生相同结果，但 FP16/BF16 visual context 的第一个不同计算结果出现在 `model.clip_embedding`，随后传递到 cross-attention 的 image K/V；block 0 self-attention 和 text K/V 在此之前完全一致。
- 按“双卡路径为基准”的决定，保留双卡 FSDP 行为不变；单卡实验路径在每次 DiT forward 前递归地将浮点 tensor 输入转换为模型参数 dtype BF16，精确复现根 FSDP 的输入转换规则。该行为由 `EngineConfig.cast_dit_forward_inputs` 控制，默认关闭，仅单卡实验入口启用，因此不会改变现有双卡或其他调用方。
- 修正后的单卡 trace 为 `experiment_logs/single_gpu/101-20260831-113743.log`。与双卡日志自动逐项比较：single stages=63、dual stages=63、common stages=63、differences=0；从原始输入、`clip_embedding`、cross image K/V、全部已埋点计算结果直到 `model.output` 均逐字节一致。
- 修正后的单卡首步 latent 为 mean=0.001893205、std=0.961710393、SHA-256=`f6845ee32b24e01bb80b8f6dfa3467c62119bb3014ef94f65718a40bd8085261`，与双卡完全相同；旧单卡 SHA-256 `26517d3b...` 的原因由此确定，不是 FSDP 权重 flatten/all-gather、矩阵 kernel 或累加顺序差异。该 probe exit code 0，GPU 2 的 NVML 峰值为 39953 MiB。
- 进一步使用实际 conditioning cache、实际 `img_emb` 和 block 0 image K/V 权重建立 FP32 参考，逐层比较 FP16 输入路径与入口先转 BF16 的路径；日志为 `experiment_logs/numerical/clip-context-dtype-20260831.log`。当前 CLIP context 全部 finite，绝对值最大 10.65625，远低于 FP16 上限，不存在需要 BF16 指数范围避免溢出的情形。
- `img_emb` 的实际 dtype 路径为：首个 LayerNorm 将 FP16/BF16 输入直接提升为 FP32，Linear 输出 BF16，GELU 保持 BF16，第二个 Linear 输出 BF16，最后 LayerNorm 输出 FP32；随后 context slice 保持 FP32，image K/V Linear 输出 BF16，K 的 RMSNorm 内部 FP32 累积后返回 BF16，FlashAttention 使用 BF16。该路径不存在 FP16 乘加或某个算子因 BF16 而获得更高计算精度。相对 FP32 参考，FP16 输入路径在 `img_emb` 每一层以及 block 0 image K/V 的 RMSE/MAE 均低于入口 BF16 路径；BF16 路径的作用仍是匹配 FSDP，而不是提高算术精度。
- 最终完整验证使用物理 GPU 2、在线 CLIP、单卡完整 BF16 DiT、DiT forward 入口 BF16 转换和 VAE 阶段 7-block offload；日志为 `experiment_logs/single_gpu/101-20260831-124017.log`，输出为 `experiment_outputs/single_gpu/101-20260831-124017.mp4`。在线 CLIP context 与缓存逐元素完全一致；4 个 segment、24 个 diffusion step、4 次 VAE decode、3 次 history encode、4 次 offload/reload 和 driving audio mux 全部完成，任务 exit code 0。
- 最终单卡 MP4 与双卡基准 `experiment_outputs/fsdp_baseline/101-20260828-152421.mp4` 均为 19587416 bytes，文件 SHA-256 均为 `7039c5f231eb64b544c4aa288ea5107411c9e7f51bdcf4c93d125d6e1610680a`；抽取后的 H.264 stream SHA-256 均为 `67357588212414de1be5895ab8fea06c886de3272562ae4cd383840e9bf36b17`，AAC/ADTS stream SHA-256 均为 `b2645001da50013ddb0cfb68115ad02a646deb397f043248d76e9766d740f046`。这确认入口 dtype 是此前单/双卡输出不一致的充分原因；对齐后完整容器逐字节一致。

## 36. 单卡 DiT 阶段 VAE CPU residency 实验（2026-08-31）

- 新增可选 `--vae-cpu-during-dit`，仅允许显式单卡非 FSDP 配置启用。VAE 在 reference encode 后移到 CPU；每个 segment 的 pose encode 前短暂回 GPU、完成后再次移到 CPU；diffusion 完成且 7 个 DiT block 已 offload 后，VAE 回 GPU执行 decode/history encode，完成后再回 CPU。VAE 的 model、mean/std 和 scale 同步迁移，并在每次迁移后校验参数 device。
- 首步 probe 日志为 `experiment_logs/single_gpu/101-20260831-130615.log`。VAE 参数为 484.1 MiB，DiT 前 CUDA allocated 精确减少 484.1 MiB；block 0 K RoPE phase peak 从未 offload probe 的 37597.6 MiB 降至 37113.5 MiB。首步 NVML 峰值从 39953 降至 39527 MiB，减少 426 MiB；latent SHA-256 仍为 `f6845ee32b24e01bb80b8f6dfa3467c62119bb3014ef94f65718a40bd8085261`。
- 完整日志为 `experiment_logs/single_gpu/101-20260831-130725.log`，输出为 `experiment_outputs/single_gpu/101-20260831-130725.mp4`。4 个 segment、24 个 diffusion step、4 次 pose encode、4 次 VAE decode、3 次 history encode、7-block DiT offload/reload 和 online CLIP 均成功，进程 exit code 0。
- 完整任务 NVML 峰值从未移动 VAE 的 40441 降到 39949 MiB，减少 492 MiB；40 GiB GPU 的物理余量从 519 增加到 1011 MiB。前三个长 segment 的 DiT 峰值均为 39949 MiB，最后一个短 segment 为 37711 MiB；VAE/history 区间最高为 38439 MiB。最终峰值仍由 DiT 决定。
- VAE 共发生 17 次迁移（包含 reference 后首次 offload），累计 4.647 秒，其中 reload 2.190 秒、offload 2.457 秒。请求 started-to-finished 为 439.766 秒，未移动 VAE 的对照为 437.209 秒，本次单次运行增加 2.557 秒；音频 mux 和一次 5.027 秒 DiT block 冷态 reload 存在波动，因此不能把差值完全归因于 VAE 迁移。最终 RSS 为 36600.2 MiB，比对照高约 2067.6 MiB，表明反复 CPU/GPU 迁移会提高 host allocator 高水位，但服务器主内存当前足够。
- 新输出与未移动 VAE 的单卡输出及原双卡基准均为 19587416 bytes，MP4 SHA-256 均为 `7039c5f231eb64b544c4aa288ea5107411c9e7f51bdcf4c93d125d6e1610680a`。迁移不改变最终结果。虽然物理余量约翻倍，但 1011 MiB 仍仅占总显存约 2.47%，尚不足以视为稳健生产余量；后续仍需继续降低 DiT attention/RoPE 峰值或做重复压力测试。

## 37. 单卡预处理 buffer 与 Self Q/K/V 提前释放（2026-08-31）

- segment 预处理完成紧凑 DiT 输入后，立即释放 `pose_segment`、`smpl_render_video`、`driving_mask_segment` 和 `null_noisy_mask`，合计约 653.9 MiB。单卡 VAE-offload 路径在 diffusion 前归还对应 allocator cache；普通 FSDP 路径不新增 cache flush。
- Self-attention 中 Q 完成 RoPE 后释放原始 Q，K 完成 RoPE 后释放原始 K；FlashAttention 返回后释放 `q_rope`、`k_rope` 和 V。每份 Q/K/V 约 476.9 MiB，计算顺序和数值路径不变。
- probe 日志为 `experiment_logs/single_gpu/101-20260831-132638.log`。block 0 最高 phase peak 从仅 VAE CPU 的 37113.5 MiB 降至 36278.3 MiB，减少 835.2 MiB；整次 probe NVML 峰值从 39527 降至 38867 MiB，减少 660 MiB。最高点已从 K RoPE 转移到 Self/Cross QKV 和 FFN residual 附近。首步 latent SHA-256 仍为 `f6845ee32b24e01bb80b8f6dfa3467c62119bb3014ef94f65718a40bd8085261`。
- 完整日志为 `experiment_logs/single_gpu/101-20260831-132759.log`，输出为 `experiment_outputs/single_gpu/101-20260831-132759.mp4`。4 个 segment、24 个 diffusion step、在线 CLIP、VAE CPU residency、7-block VAE 阶段 offload/reload 和音频 mux 全部成功，exit code 0。
- 完整 NVML 峰值为 39249/40960 MiB，比仅 VAE CPU 对照的 39949 MiB 再降 700 MiB，比未移动 VAE 的 40441 MiB 降 1192 MiB；物理余量达到 1711 MiB（4.18%）。请求用时 436.526 秒，未观察到可归因于提前释放的性能损失。
- 最终 MP4 为 19587416 bytes，SHA-256=`7039c5f231eb64b544c4aa288ea5107411c9e7f51bdcf4c93d125d6e1610680a`，与双卡基准逐字节一致。下一批局部高点已接近：Self/Cross QKV 与 FFN residual 均约 36278 MiB，继续降低峰值需要同时处理这些阶段，而不是只优化 RoPE。
