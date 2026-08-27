# SCAIL2 模型加载优化与故障排查报告

日期：2026-08-27  
状态：双卡 FSDP 加速加载已完成端到端验证

## 1. 结论摘要

本轮工作完成了 SCAIL2 双卡 FSDP 实验入口、初始化阶段计时、T5/DiT 加速加载、完整推理回归以及推理“卡死”根因定位。

最终结论如下：

- T5 和 DiT 均可通过 meta device 构建、checkpoint 直接 `assign=True` 的方式跳过无效参数初始化和大规模参数复制。
- T5 checkpoint 通过 `torch.load(weights_only=True, mmap=True)` 打开，DiT safetensors 直接绑定到 meta 模型。
- 双卡 `engine_load` 从未加速完整成功基线的 **282.022 秒**降至 **17.862 秒**，减少 **264.160 秒（93.67%）**，加载阶段约 **15.79 倍加速**。
- 加速加载前后的推理耗时基本一致：ready 到任务完成分别为 439.626 秒和 437.707 秒，差异仅 **1.919 秒（0.44%）**，说明加载优化没有改变稳定态推理性能。
- 初始化加推理的总时间从约 **721.648 秒（12 分 01.6 秒）**降至 **455.569 秒（7 分 35.6 秒）**，单任务冷启动总耗时减少约 **4 分 26 秒（36.9%）**。
- 此前多次表现为“一张卡 0%、另一张卡 100%”的推理卡死，不是 meta-assign 引入的副作用。首发故障是 rank 0 在 DiT 第三个 sampling step 的 FFN/GELU 中因 CUDA allocator 碎片化发生 OOM；rank 1 随后等待 rank 0 未加入的 FSDP all-gather，形成次生 NCCL 停顿。
- 启用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 后，标准加载和加速加载均能完成相同输入的完整四段推理。

## 2. 实验范围与固定条件

- 服务器：单机 8×NVIDIA A100 40 GB，GPU 间为 NVLink/NVSwitch 全互联。
- 执行环境：宿主机 Conda 环境 `scail2-single-gpu`，不经过容器。
- 模型目录：`/raid/scail-2-20260819`。
- 当前验证使用物理 GPU 2、3；GPU 4、5 因历史硬件错误记录不再用于实验。
- 固定测试数据：`testdata/101`，输入为 512×896、30 fps、297 帧。
- 执行模式：双卡 FSDP，T5 和 DiT 均启用 FSDP；DiT 为 40 个 Transformer block，6 个 sampling step。
- 命令行入口：[run_fsdp_experiment.py](run_fsdp_experiment.py)。该入口绕过外部服务，负责路径校验、GPU 绑定、torchrun 启动、时间戳日志和输出管理。

## 3. 初始问题与基线

初始标准加载路径会先在 CPU 上完整构建模型并初始化参数，然后加载 checkpoint 覆盖这些参数。对推理 checkpoint 而言，随机初始化和后续复制没有业务价值，却产生了大量 CPU 计算、内存分配和数据复制。

最早的 init-only 基线日志 [101-20260827-133218.log](experiment_logs/fsdp_baseline/101-20260827-133218.log) 中：

- `engine_load`：289.269 秒。
- T5 加载/FSDP：关键 rank 约 79.668 秒。
- DiT 空模型构建及 BF16 转换：约 178.546 秒。
- DiT checkpoint 复制：约 5.728 秒。
- DiT FSDP wrap：约 11.790 秒。

另一次稳定标准路径测得 `engine_load=276.538` 秒。最终用于完整推理对照的标准加载成功实验 [101-20260827-161806.log](experiment_logs/fsdp_baseline/101-20260827-161806.log) 为 282.022 秒。不同运行之间存在文件缓存和系统状态波动，但共同结论是：启动耗时约 4.5--5 分钟，主要瓶颈是 T5/DiT 的普通模型构建与加载。

## 4. 优化实现

### 4.1 DiT meta-assign

DiT 优化路径执行以下操作：

1. 在 `torch.device("meta")` 上构建 `SCAIL2Model` 结构，只创建参数元数据，不分配真实存储，也不执行有成本的真实参数初始化。
2. 使用 safetensors 读取 BF16 checkpoint。
3. 验证 checkpoint tensor 数量、dtype 和参数量。
4. 通过 `load_state_dict(strict=True, assign=True)` 将 checkpoint tensor 直接绑定到模型参数，避免复制到预先分配的参数存储。
5. 检查 FSDP 前不存在残留 meta parameter/buffer。

DiT 中的 RoPE `freqs` 不属于 checkpoint，也没有注册为 parameter/buffer。meta 构建时它同样位于 meta device，因此加载后必须在 CPU 上调用 `_make_freqs()` 重新物化，并额外检查 `freqs.is_meta == False`。这不是推理算法修改，而是保证 meta 加载后恢复标准路径原本存在的非 checkpoint 张量。

单独启用 DiT 优化后：

- DiT 构建从最多 181.140 秒降至约 0.169 秒。
- `engine_load` 从 276.538 秒降至 84.962 秒。
- 剩余最大瓶颈转为未优化的 T5，约 70 秒。

对应日志：[101-20260827-141715.log](experiment_logs/fsdp_baseline/101-20260827-141715.log)。

### 4.2 T5 meta-assign 与 mmap

T5 优化路径执行以下操作：

1. 在 meta device 上构建 UMT5-XXL encoder。
2. 使用 `torch.load(..., map_location="cpu", weights_only=True, mmap=True)` 打开 checkpoint。
3. 使用 `load_state_dict(strict=True, assign=True)` 直接绑定参数。
4. 删除临时 state dict，并在 FSDP 前检查没有残留 meta tensor。
5. 保持原有 `sync_module_states=False` 的 T5 FSDP 行为。

T5 总阶段由约 70 秒降至约 2.94--3.16 秒。T5 和 DiT 同时优化后的首次 init-only 成功日志 [101-20260827-142145.log](experiment_logs/fsdp_baseline/101-20260827-142145.log) 测得 `engine_load=16.511` 秒。

需要注意：日志中的 mmap checkpoint read 约 0.03 秒仅代表建立映射和读取序列化元数据，并不等价于 11 GB 权重已经全部从存储读入物理内存。真实 page fault 和读取成本会部分发生在 FSDP wrap、同步或首次访问阶段。

### 4.3 安全开关与可观测性

- `EngineConfig` 新增 `t5_meta_load` 和 `dit_meta_load`，默认均为 `False`，避免无意改变其他入口的默认行为。
- 实验 CLI 显式同时启用两项优化，并在日志头记录实际加载模式。
- 初始化日志细分为 process group、T5、VAE、CLIP、DiT 构建、checkpoint read/validate/assign、FSDP wrap、barrier 和 ready。
- 每次实验自动保存完整 stdout/stderr，所有落盘记录带本地时区和毫秒时间戳。
- 增加可选 `--diagnose-fsdp`：启用 rank/block/step 埋点、显存 allocated/reserved、NCCL flight recorder 信息和 120 秒 process-group watchdog。
- 将旧 `torch.cuda.amp.autocast` 全部迁移为 `torch.amp.autocast("cuda", ...)`，消除 AMP deprecation `FutureWarning`。

## 5. 卡死现象及排查过程

### 5.1 初始现象

T5/DiT meta 加载的 init-only 测试成功，但多次完整推理在第 1 个 segment 的第 3 个 sampling step 附近不再前进：

- 一张 GPU 利用率降到 0%，另一张保持约 100%。
- 两个 worker 仍存活，显存仍被占用。
- 没有立即出现清晰的 Python traceback。
- 最初使用的 GPU 5 存在历史 Xid/NVLink replay/CRC 错误，因此一度怀疑 GPU 或 NVLink 故障。

### 5.2 被逐步排除的假设

| 假设 | 排查方式 | 结论 |
|---|---|---|
| GPU 5 硬件问题 | 切换到 NVLink error counter 为 0 的 GPU 2、3 | 同一位置复现，不是必要条件 |
| meta-assign 改变了推理行为 | 运行标准/混合加载组合 | 标准加载也能复现，meta 不是必要条件 |
| 新增 `torch.cuda.synchronize()` 的副作用 | 删除 sampling 路径中的临时同步点 | 仍在 step 3 复现 |
| 近期源代码修改 | 完整恢复到 13:13 成功时的 legacy 代码 | 仍复现 |
| 原服务器临时状态 | 在另一台配置相同的服务器运行 | 同样失败，非单机特有状态 |
| 单纯 NCCL 或 FSDP 死锁 | 加入 block 级事件、显存记录和 watchdog | 发现 rank 0 先发生 OOM，NCCL 等待是后果 |

并行运行三组加载模式曾用于快速缩小范围，但共享主机资源引入了额外变量；后续改为单作业顺序实验。完整回滚和跨机器复现尤其重要，因为它们证明“最近的 meta 代码导致卡死”和“原机器坏了”都不足以解释现象。

### 5.3 根因定位

根因日志：[101-20260827-160920.log](experiment_logs/fsdp_baseline/101-20260827-160920.log)。

诊断显示：

1. 两个 rank 都完成 sampling step 1、2。
2. step 3 中，rank 0 完成 block 0 后进入 block 1；rank 1 已继续进入后续 block，rank 执行序列开始分叉。
3. rank 0 在 `wan/modules/model_scail2.py` 的 block 1 FFN/GELU 中申请 1.26 GiB 失败并抛出 `torch.OutOfMemoryError`。
4. OOM 时 GPU 仍显示约 1.15 GiB 物理空闲；PyTorch 已分配 32.23 GiB，同时有 5.01 GiB reserved-but-unallocated。空闲/保留空间无法满足连续分配，符合 caching allocator 碎片化。
5. rank 0 没有加入下一次 FSDP collective；rank 1 已继续提交 all-gather，因此表现为一张卡等待、另一张卡保持忙碌。
6. watchdog 最终确认 rank 0 停在 collective #302 之后，rank 1 已提交到 #304；`SeqNum=303` 的 `_ALLGATHER_BASE` 在 120 秒后超时。

因此，外部看到的“FSDP/NCCL 卡死”是 **rank 0 OOM 后的次生通信等待**。异步 CUDA 执行和多 rank 行为掩盖了首发错误，使问题长期看起来像 collective 死锁。

### 5.4 修复

在 torch worker 启动前设置：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

实验入口通过 `--expandable-segments` 设置该配置。启用后，长 segment 初始 reserved 显存由失败实验的约 38.0--38.2 GiB 降至约 29.3 GiB，运行高峰仍约 38.2 GiB，但 allocator 可以更有效地扩展和复用显存段。

标准加载修复验证 [101-20260827-161806.log](experiment_logs/fsdp_baseline/101-20260827-161806.log) 完成全部 4 个 segment、每段 6 步，没有 OOM、rank 分叉或 NCCL timeout。

## 6. 最终完整回归

定位根因后，重新恢复 T5/DiT meta-assign、初始化计时和 AMP warning 修复，并依次完成：

- init-only 回归：[101-20260827-164018.log](experiment_logs/fsdp_baseline/101-20260827-164018.log)，`engine_load=17.828` 秒。
- 日志/AMP 回归：[101-20260827-164502.log](experiment_logs/fsdp_baseline/101-20260827-164502.log)，`engine_load=17.821` 秒，无 AMP `FutureWarning`。
- 完整推理回归：[101-20260827-164739.log](experiment_logs/fsdp_baseline/101-20260827-164739.log)，`engine_load=17.862` 秒，任务成功退出。

最终执行命令：

```bash
/home/panminghao/miniconda3/envs/scail2-single-gpu/bin/python -u \
  run_fsdp_experiment.py --physical-gpus 2,3 --expandable-segments
```

输出：[101-20260827-164739.mp4](experiment_outputs/fsdp_baseline/101-20260827-164739.mp4)。`ffprobe` 验证结果为：

- H.264 视频、AAC 音频。
- 512×896、30 fps、297 帧、9.9 秒。
- 文件大小 19,587,416 bytes。

该规格与标准加载成功输出一致。这里验证的是端到端可运行性和媒体结构一致性；尚未进行逐帧数值误差或感知质量对比，因此不能仅凭相同文件大小宣称两次输出逐字节一致。

## 7. 性能对比

### 7.1 加载与端到端时间

| 指标 | 标准加载成功基线 | meta-assign 加速加载 | 变化 |
|---|---:|---:|---:|
| Engine 加载 | 282.022 s | 17.862 s | -264.160 s，-93.67% |
| 推理请求（ready 到完成） | 439.626 s | 437.707 s | -1.919 s，-0.44% |
| 初始化 + 推理 | 721.648 s | 455.569 s | -266.079 s，-36.9% |

标准加载成功基线启用了高频 FSDP 诊断，而最终加速加载回归未启用，因此推理部分 1--2 秒的差异不能归因于 meta-assign。采样步耗时基本相同：前三个 81 帧 segment 均约 17.4--17.5 秒/步，最后一个 57 帧 segment 均约 10.8 秒/步。

### 7.2 最终加载阶段细分

以下取两个 rank 中的较慢值：

| 阶段 | 耗时 |
|---|---:|
| Process group | 2.749 s |
| Pipeline 前置初始化（尚未细分） | 约 4.58 s |
| T5 总阶段 | 3.122 s |
| VAE | 0.290 s |
| CLIP | 1.450 s |
| DiT meta 构建 | 0.166 s |
| DiT checkpoint 聚合阶段 | 0.228 s |
| DiT FSDP 前 barrier | 0.212 s |
| DiT FSDP wrap/同步 | 5.051 s |
| Pipeline load 总计 | 15.112 s |
| Engine load 总计 | 17.862 s |

当前剩余最大的启动区段是 DiT FSDP wrap/同步约 5.05 秒、尚未细分的 pipeline 前置 import/config/provenance 初始化约 4.58 秒，以及 T5 约 3.12 秒。

## 8. 当前状态、已知问题与建议

### 8.1 当前可用状态

- T5/DiT meta-assign 已通过 init-only 和完整推理验证。
- `expandable_segments` 是当前 40 GB A100 双卡 FSDP 完整推理的必要稳定性配置，后续完整实验不应遗漏。
- `--diagnose-fsdp` 保留为故障模式开关；日常性能实验应关闭，以避免大量日志干扰。
- 实验日志和输出目录已加入 `.gitignore`，但文件仍保留在本地用于复核。
- GPU 2、3 在最终实验后均已释放，无残留 worker。

### 8.2 已知但未阻塞的问题

- FSDP 对象销毁时仍会输出 `PyInterpreter.cpp` weak-reference warning；它发生在结果保存和任务成功之后，不影响本轮正确性，但后续可单独调查资源释放顺序。
- `ffmpeg` 输出过一次 multiple `-r` warning，不影响最终 30 fps 文件，但可清理重复帧率参数。
- 受限沙箱内 torchrun 的本地 TCPStore 曾无法解析/连接 `localhost`；多 rank GPU 实验需要在允许本机 rank 通信的宿主执行环境运行。
- 当前 17.8 秒结果可能受 Linux page cache 影响。若要评估真实冷启动，应在可控条件下区分冷缓存、热缓存并重复多次，报告 P50/P95。
- 当前工作证明媒体规格与执行流程一致，但若要把优化合入正式版本，还应增加固定 seed 的逐帧/特征级质量对比。

### 8.3 下一步建议

1. 将 expandable allocator 设为实验入口的安全默认值，只有专门做 allocator A/B 时才关闭。
2. 保留低噪声的 step/rank watchdog，在任一 rank 首发异常时尽快终止整个 process group，避免再次留下表面“卡死”的孤儿进程。
3. 继续细分约 4.58 秒的 pipeline 前置阶段，分别测量 Python import、配置加载和 checkpoint provenance。
4. 如仍需优化冷启动，优先研究 DiT FSDP wrap 的 5.05 秒、T5 tokenizer 的约 1.37 秒和 T5 FSDP wrap 的约 1.62 秒。
5. 在加载路径稳定后回到主目标：单卡推理可行性。先只驻留 DiT，测量 81 帧 segment 的真实 allocated/reserved 峰值；必要时缓存 T5 embedding，并对 CLIP/VAE 做分阶段 offload。
6. 进入吞吐量实验前固定代码版本、模型哈希、缓存状态、GPU 对和 allocator 配置，避免再次把环境差异误判为模型加载副作用。

## 9. 关键证据索引

- 完整实验上下文：[SCAIL2_EXPERIMENT_CONTEXT.md](SCAIL2_EXPERIMENT_CONTEXT.md)
- 初始 init-only 基线：[101-20260827-133218.log](experiment_logs/fsdp_baseline/101-20260827-133218.log)
- DiT meta 初始化：[101-20260827-141715.log](experiment_logs/fsdp_baseline/101-20260827-141715.log)
- T5+DiT meta 初始化：[101-20260827-142145.log](experiment_logs/fsdp_baseline/101-20260827-142145.log)
- OOM/NCCL 根因日志：[101-20260827-160920.log](experiment_logs/fsdp_baseline/101-20260827-160920.log)
- 标准加载+expandable 成功基线：[101-20260827-161806.log](experiment_logs/fsdp_baseline/101-20260827-161806.log)
- 最终加速加载完整回归：[101-20260827-164739.log](experiment_logs/fsdp_baseline/101-20260827-164739.log)
- 最终输出：[101-20260827-164739.mp4](experiment_outputs/fsdp_baseline/101-20260827-164739.mp4)
