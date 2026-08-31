# SCAIL2 单卡峰值显存进一步优化报告

日期：2026-08-31

## 1. 结论

在旧双卡兼容配置下，单卡完整推理的 NVML 峰值为 40441/40960 MiB，只剩 519 MiB。今天通过 VAE residency 和 tensor 生命周期优化先将峰值降到 39249 MiB；切换到新的不转换标准并使用 CLIP cache 后，最新完整回归峰值进一步降到 38941 MiB。

1. DiT 执行期间把 VAE 放在 CPU，需要 VAE 时再短暂迁回 GPU。
2. 提前释放 segment 预处理 buffer，以及 Self-attention 中已经完成最后一次使用的原始 Q/K/V。

相对初始单卡基线累计降低 1500 MiB，物理余量从 519 MiB 增加到 2019 MiB（4.93%）。最新请求耗时 434.250 秒；DiT 每步性能没有出现可识别的退化。

当前峰值已从单一 K RoPE 阶段转移到 Self QKV、Cross QKV 和 FFN residual 一组接近的阶段。继续优化需要降低多个共同高点，工作重点不再是单独处理 RoPE。

## 2. 基线与约束

本轮初始基线采用复现旧双卡 FSDP 语义的完整单卡回归：

```text
experiment_logs/single_gpu/101-20260831-124017.log
```

| 指标 | 基线 |
|---|---:|
| GPU | A100-SXM4-40GB，物理 GPU 2 |
| 完整 DiT | 40 个 BF16 block 全部常驻 GPU |
| DiT CPU offload | 不允许 |
| VAE 阶段 | 已卸载末尾 7 个 DiT block |
| 在线 CLIP | reference 阶段执行后 offload |
| NVML 峰值 | 40441 MiB |
| 物理余量 | 519 MiB（1.27%） |

优化继续遵守以下约束：DiT diffusion 期间不能按 block 往返 CPU；CPU 中保留一份完整 DiT master；VAE、CLIP 等非 DiT 模型允许按阶段迁移；所有修改必须保持最终输出一致。

## 3. VAE 在 DiT 阶段驻留 CPU

VAE 参数约为 484.1 MiB。原流程即使正在执行 DiT，VAE 权重仍保留在 GPU。新增 `--vae-cpu-during-dit` 后，设备切换顺序为：

```text
reference encode 完成 -> VAE 到 CPU
pose encode 前        -> VAE 到 GPU
pose encode 后        -> VAE 到 CPU
diffusion 完成        -> 卸载 7 个 DiT block，VAE 到 GPU
decode/history 完成   -> VAE 到 CPU，恢复 7 个 DiT block
```

VAE 的 model、mean/std 和 scale 一起迁移，每次迁移后检查参数所在设备。该功能只允许单卡非 FSDP 配置显式启用。

### 3.1 Probe 结果

日志：`experiment_logs/single_gpu/101-20260831-130615.log`

| 指标 | 对齐后基线 | VAE CPU | 变化 |
|---|---:|---:|---:|
| DiT 前 allocated | — | 精确减少 484.1 MiB | -484.1 MiB |
| block 0 K RoPE peak | 37597.6 MiB | 37113.5 MiB | -484.1 MiB |
| Probe NVML 峰值 | 39953 MiB | 39527 MiB | -426 MiB |

首步 latent SHA-256 保持为 `f6845ee32b24e01bb80b8f6dfa3467c62119bb3014ef94f65718a40bd8085261`。

### 3.2 完整回归结果

日志：`experiment_logs/single_gpu/101-20260831-130725.log`

| 指标 | 对齐后基线 | VAE CPU | 变化 |
|---|---:|---:|---:|
| 完整 NVML 峰值 | 40441 MiB | 39949 MiB | -492 MiB |
| 物理余量 | 519 MiB | 1011 MiB | +492 MiB |
| 请求时间 | 437.209 秒 | 439.766 秒 | +2.557 秒 |

VAE 共迁移 17 次，累计 4.647 秒，其中 reload 2.190 秒、offload 2.457 秒。总请求的单次差值还包含音频和 DiT block 冷态 reload 波动，不能全部归因于 VAE 迁移。

前三个长 segment 的 DiT 峰值均为 39949 MiB；VAE/history 区间最高为 38439 MiB。最高点仍在 DiT，说明 VAE residency 优化有效，但不足以单独提供稳定余量。

## 4. 预处理 buffer 提前释放

每个 segment 在 diffusion 前会生成较大的全分辨率或半分辨率中间 tensor。它们生成紧凑的 pose/mask latent 后不再被使用，但原先一直保活到 diffusion 结束。

| Buffer | 大小 | 最后用途 | 新释放点 |
|---|---:|---|---|
| `pose_segment` | 425.2 MiB | 生成半分辨率 pose | `F.interpolate` 返回后 |
| `smpl_render_video` | 106.3 MiB | VAE pose encode | `vae.encode` 返回后 |
| `driving_mask_segment` | 106.3 MiB | 压缩为 latent mask | mask 压缩返回后 |
| `null_noisy_mask` | 16.1 MiB | 拼接 `ref_masks` | `torch.cat` 返回后 |
| 合计 | **653.9 MiB** | — | diffusion 前 |

单卡 VAE-offload 路径在进入 diffusion 前执行 allocator cache 清理，使这些已死亡 tensor 不再计入物理 NVML 峰值。普通双卡 FSDP 路径不新增 cache flush，避免改变既有执行行为。

## 5. 原始 Q/K/V 提前释放

首个长 segment 中，Self-attention 的 Q、K、V 均为 BF16 `[1, 48832, 40, 128]`，每份约 476.9 MiB。

原生命周期：

```text
生成 Q/K/V
  -> Q RoPE
  -> K RoPE
  -> FlashAttention
  -> self-attention 返回时释放原始 Q/K
```

新生命周期：

```text
Q RoPE 返回 -> 立即释放原始 Q
K RoPE 返回 -> 立即释放原始 K
FlashAttention 返回 -> 释放 q_rope、k_rope 和原始 V
```

这些 tensor 均在最后一次使用后释放，没有改变计算顺序、dtype、RoPE、FlashAttention 或输出 projection。

## 6. 局部显存变化

优化后 probe 日志：

```text
experiment_logs/single_gpu/101-20260831-132638.log
```

| 阶段 | 仅 VAE CPU | 增加 buffer/QKV 释放 | 变化 |
|---|---:|---:|---:|
| Self 输入 phase peak | 35024.9 MiB | 34370.9 MiB | -654.0 MiB |
| Self Q/K/V phase peak | 36932.2 MiB | 36278.3 MiB | -653.9 MiB |
| K RoPE phase peak | 37113.5 MiB | 35982.7 MiB | -1130.8 MiB |
| FlashAttention phase peak | 36939.5 MiB | 35331.8 MiB | -1607.7 MiB |
| Self output projection phase peak | 36456.2 MiB | 34371.6 MiB | -2084.6 MiB |
| block 0 最高 phase peak | 37113.5 MiB | 36278.3 MiB | -835.2 MiB |
| Probe NVML 峰值 | 39527 MiB | 38867 MiB | -660 MiB |

局部变化与 tensor 大小吻合：

- 所有阶段首先少约 653.9 MiB 预处理 buffer。
- K RoPE 额外少保留约 476.9 MiB 原始 Q。
- FlashAttention 再少保留约 476.9 MiB 原始 K。
- output projection 再少保留约 476.9 MiB 原始 V。

局部最大降幅达到约 2084.6 MiB，但整个 block 只降低 835.2 MiB，因为最高点随之转移到其他阶段。

## 7. 最终完整推理结果

日志与输出：

- 日志：`experiment_logs/single_gpu/101-20260831-132759.log`
- 输出：`experiment_outputs/single_gpu/101-20260831-132759.mp4`

完整流程包含在线 CLIP、4 个 segment、24 个 diffusion step、VAE CPU residency、4 次 VAE decode、3 次 history encode、7-block DiT 阶段切换和 driving audio。

### 7.1 分阶段 NVML 峰值

| 阶段 | 峰值 MiB |
|---|---:|
| Reference prepare | 34533 |
| Online CLIP | 33083 |
| Segment 1 DiT | **39249** |
| Segment 1 VAE/history | 38439 |
| Segment 2 DiT | 38889 |
| Segment 2 VAE/history | 38479 |
| Segment 3 DiT | 38889 |
| Segment 3 VAE/history | 38479 |
| Segment 4 DiT | 37531 |
| Segment 4 VAE/history | 38259 |
| Video encode / audio publish | 31927 |

### 7.2 总体对比

| 方案 | NVML 峰值 | 物理余量 | 相对初始基线 |
|---|---:|---:|---:|
| 数值对齐后的单卡基线 | 40441 MiB | 519 MiB（1.27%） | — |
| VAE 在 DiT 期间驻留 CPU | 39949 MiB | 1011 MiB（2.47%） | -492 MiB |
| 再提前释放预处理/QKV | **39249 MiB** | **1711 MiB（4.18%）** | **-1192 MiB** |
| 新标准 + CLIP cache | **38941 MiB** | **2019 MiB（4.93%）** | **-1500 MiB** |

### 7.3 优化前后性能比较

请求时间采用日志中的 `started_at` 到 `finished_at`，不包含 engine 初始化。三组单卡实验都执行在线 CLIP、4 个 segment、24 个 diffusion step、视频编码和 driving audio。

| 指标 | 对齐后单卡基线 | VAE CPU | 最终优化 | 最终相对基线 |
|---|---:|---:|---:|---:|
| 请求总时间 | 437.209 秒 | 439.766 秒 | **436.526 秒** | **-0.683 秒（-0.16%）** |
| 81 帧 DiT 平均每步 | 约 16.69 秒 | 约 16.69 秒 | **约 16.69 秒** | 无可见变化 |
| 57 帧 DiT 平均每步 | 10.31 秒 | 10.31 秒 | **10.31 秒** | 无变化 |
| 在线 CLIP | 3.566 秒 | 1.780 秒 | 1.783 秒 | 存在冷/热态波动 |
| 7-block offload + reload | 5.974 秒 | 10.193 秒 | 6.403 秒 | +0.429 秒 |
| VAE CPU/GPU 迁移 | 0 | 4.647 秒 | 4.851 秒 | +4.851 秒 |
| Audio mux | 5.266 秒 | 2.445 秒 | 2.431 秒 | 存在外部工具波动 |

最终版本比仅启用 VAE CPU 的实验快 3.240 秒，但中间实验包含一次 5.027 秒的冷态 DiT block reload，不能把这 3.240 秒解释成提前释放带来的计算加速。更可靠的结论来自 diffusion step：三组实验的 DiT 每步耗时相同，说明预处理 buffer 和 Q/K/V 生命周期缩短没有增加模型计算成本。

VAE residency 会新增约 4.7--4.9 秒显式迁移，最终端到端时间却没有同比增加，因为 CLIP、block reload、视频编码和 audio mux 都存在数秒级单次波动。因此当前数据足以判断“没有明显性能退化”，但不足以声称端到端性能提高；若需要精确量化，应对每个方案做多次热态重复运行并取中位数。

### 7.4 与双卡 FSDP 的性能比较

双卡对照使用 `experiment_logs/fsdp_baseline/101-20260828-152421.log`。它和最新单卡都使用缓存 CLIP context，因此 conditioning 获取方式基本一致；两者的剩余区别是双卡仍采用旧 FSDP 入口 BF16 转换，而最新单卡采用新的 FP16 context 标准。

| 指标 | 双卡 FSDP | 新标准单卡 + CLIP cache | 单卡相对双卡 |
|---|---:|---:|---:|
| 使用 GPU 数 | 2 | **1** | -1 张 GPU |
| 请求总时间 | 424.714 秒 | **434.250 秒** | +9.536 秒（+2.25%） |
| 81 帧 DiT 每步 | 16.91--16.99 秒 | **约 16.71 秒** | 单卡快约 1.2%--1.6% |
| 57 帧 DiT 每步 | 10.49--10.53 秒 | **10.32 秒** | 单卡快约 1.6%--2.0% |

单卡 DiT 本身略快，端到端多出的时间主要来自 VAE 迁移和阶段间权重恢复。最新单卡和双卡都使用缓存 CLIP context，conditioning 口径已基本一致；单卡只用一张 GPU，而请求延迟比旧双卡高 2.25%。在 CPU 内存和 I/O 足够、两张卡分别运行独立 worker 的前提下，单位 GPU 吞吐仍明显优于双卡 FSDP。

### 7.5 新标准 + CLIP cache 性能回归

最新回归命令使用完整单卡优化组合，但不启用 `--cast-dit-forward-inputs`，并通过 `--cached-clip` 完全跳过 CLIP 模型加载和在线编码：

```text
python run_single_gpu_experiment.py \
  --full-inference \
  --physical-gpu 2 \
  --cached-clip \
  --vae-cpu-during-dit
```

日志为 `experiment_logs/single_gpu/101-20260831-140335.log`，输出为 `experiment_outputs/single_gpu/101-20260831-140335.mp4`。

| 指标 | 在线 CLIP + BF16 入口 | CLIP cache + FP16 context | 变化 |
|---|---:|---:|---:|
| Engine load | 17.835 秒 | **12.329 秒** | -5.506 秒 |
| 请求总时间 | 436.526 秒 | **434.250 秒** | -2.276 秒（-0.52%） |
| 81 帧 DiT 平均每步 | 约 16.69 秒 | **约 16.71 秒** | +0.02 秒，运行波动 |
| 57 帧 DiT 平均每步 | 10.31 秒 | **10.32 秒** | +0.01 秒，运行波动 |
| 7-block offload + reload | 6.403 秒 | **6.362 秒** | -0.041 秒 |
| VAE CPU/GPU 迁移 | 4.851 秒 | **4.632 秒** | -0.219 秒 |
| Audio mux | 2.431 秒 | **2.444 秒** | +0.013 秒 |
| NVML 峰值 | 39249 MiB | **38941 MiB** | -308 MiB |
| 进程最终 RSS | 36710.1 MiB | **34096.6 MiB** | -2613.5 MiB |

CLIP cache 的主要确定性收益是跳过 CLIP checkpoint 加载，使 engine load 缩短 5.506 秒并降低 host RSS。请求总时间减少 2.276 秒，但其中仍混有正常运行波动；DiT 每步时间几乎不变，说明 FP16 context 标准和 cached CLIP 均未对核心 diffusion 性能造成实质影响。

新输出为 H.264 512×896、297 帧、30 fps，并包含 9.9 秒 AAC 音频；文件大小 19652849 bytes，SHA-256=`fcb0871b57305440b8cd33ab8e3960a7d8f81f68c401190addda80ac59137d7e`。它与历史“不转换 + 在线 CLIP”完整输出逐字节一致，因此这次性能测试同时完成了新数值标准的回归。

旧 BF16 对齐实验的 MP4 为 19587416 bytes，SHA-256 为：

```text
7039c5f231eb64b544c4aa288ea5107411c9e7f51bdcf4c93d125d6e1610680a
```

它与旧双卡 FSDP 基准逐字节一致，说明两轮显存优化没有改变该实验配置的结果；当前采用的新 FP16 标准输出见第 7.5 节。

## 8. 当前判断与下一步

本轮修改应保留：VAE 阶段 residency 和 tensor 生命周期缩短都符合“DiT 运行期间完整权重常驻 GPU”的约束，并已经完成结果回归。

但 2019 MiB 只占 40 GiB 的 4.93%，比此前安全，但尚不能直接视为稳健生产余量。生产化前仍应做重复请求、冷启动、不同输入长度和 allocator 碎片压力测试。

当前 probe 中约 36278 MiB 的最高点同时出现在 Self QKV、Cross QKV 和 FFN residual 附近。下一步若继续降低峰值，需要处理这些共享高点，例如缩短后续 block FP32 residual 与 QKV/RMSNorm 临时量的重叠；仅继续压缩 K RoPE 已无法显著降低全局峰值。
