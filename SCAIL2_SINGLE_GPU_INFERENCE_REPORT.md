# SCAIL2 单卡推理改造与验证报告

日期：2026-08-28

> 2026-08-31 更新：本文主体记录的是单卡化的初始实验版本。当前持久服务在
> READY 状态保持完整 DiT、VAE、CLIP 位于 GPU；每个 segment 只在 DiT
> diffusion 期间卸载 VAE，VAE 迁移由 17 次降为 8 次，运行时改用 T5-only
> 文件接口，并已通过同一 engine 连续两次完整推理。最新数据见
> `SCAIL2_EXPERIMENT_CONTEXT.md` 第 39 节。

## 1. 结论

SCAIL2 已完成从双卡 FSDP 到单卡推理的实验性改造，并在一张 A100-SXM4-40GB 上跑通完整 297 帧流程。

当前方案满足两个核心要求：

- DiT 执行 diffusion 时，40 个 block 的 BF16 权重全部位于 GPU，不在 block 间做 CPU offload。
- VAE decode 和 history encode 前，临时卸载末尾 7 个 DiT block；阶段结束后再从 CPU master 恢复完整 DiT。
- T5 context 继续使用缓存；CLIP 在 reference 阶段短暂上 GPU，生成 visual context 后立即回 CPU。

最新完整单卡推理包含在线 CLIP 和 driving audio，耗时 429.555 秒；双卡 FSDP 对照为 424.714 秒。单卡慢 1.14%，但只使用一张 GPU，因此在两张相同 GPU 上运行两个独立单卡 worker 时，理论并发吞吐仍接近双卡 worker 的两倍。

单卡方案目前最大的限制是 DiT 峰值达到 40081/40960 MiB，只剩约 879 MiB 物理余量。CLIP reference 阶段可以短暂叠加，但在 diffusion 峰值期间不能再增加较大的 CUDA 常驻 buffer。

## 2. 实验目标与条件

本阶段目标不是把 DiT 本身分层卸载，而是验证完整 DiT 能否在单卡上持续执行，并只在非 DiT 阶段进行权重切换。

| 项目 | 配置 |
|---|---|
| GPU | 物理 GPU 2，A100-SXM4-40GB |
| 模型目录 | `/raid/scail-2-20260819` |
| 测试数据 | `testdata/101` |
| 输出规格 | 512×896，297 帧，30 fps |
| Segment | 81、81、81、57 帧 |
| Diffusion | 每段 6 步，共 24 步 |
| DiT | BF16，40 个 block，全量单卡运行 |
| Conditioning | T5 使用预计算缓存；CLIP 在线编码后 offload |
| FFN/RoPE 分块 | 8192 / 8192 |
| CUDA allocator | `expandable_segments:True` |
| 单卡入口 | `run_single_gpu_experiment.py` |

单卡入口不启动 `torchrun`，不初始化 process group，也不创建 FSDP wrapper，从而避免在实验路径中残留双卡生命周期和 collective 逻辑。

## 3. 单卡执行结构

CPU 内存中始终保留一份完整 DiT checkpoint tensor，约 31272 MiB。它既是权重的 host master，也是 VAE 阶段结束后恢复 CUDA 参数的数据源。

```text
初始化
  CPU master：完整 40-block DiT
  CPU：CLIP visual encoder
  GPU：完整 40-block DiT
             │
             ▼
Reference：CLIP 短暂上 GPU生成 context，随后回 CPU
             │
             ▼
DiT diffusion：40 个 block 全部在 GPU 执行
             │
             ▼
卸载 blocks 33--39：GPU 释放 5391.8 MiB
             │
             ▼
VAE decode + history encode
             │
             ▼
从 CPU master 恢复 blocks 33--39
             │
             └──────── 下一 segment
```

卸载时不会先把 CUDA 权重复制回 CPU，而是直接用已经存在的 CPU master 参数替换对应 CUDA 参数。这样 offload 本身只负责释放 GPU 权重；reload 才产生 CPU 到 GPU 的传输。

## 4. 为什么选择卸载 7 个 block

40 个 DiT block 的 checkpoint 大小完全相同，每个为 770.261 MiB。完整 DiT 权重为 31272.0 MiB，其中非 block 参数约 461.6 MiB。

| 方案 | 释放的 DiT 权重 | VAE 实测结果 |
|---|---:|---|
| 不卸载 | 0 | 完整 DiT 与 VAE 激活无法安全叠加 |
| 卸载 7 个 block | 5391.8 MiB | VAE 峰值 38470.8 MiB，余量 2489.2 MiB |

因此当前固定卸载末尾 `blocks.33-39`。这不是理论最小值，而是在 40GB 卡上兼顾安全余量和 reload 开销的实验配置。

## 5. 分阶段显存占用

下表中的 `device used` 是 CUDA/NVML 口径，包含 PyTorch reserved、CUDA context、kernel workspace 和驱动分配；它比 `allocated` 更接近 GPU 监控程序显示的数字。

| 阶段 | GPU 上的 DiT | allocated | reserved | device/NVML used | 物理余量 |
|---|---|---:|---:|---:|---:|
| Engine ready | 40 blocks | 31756.1 | 31768.0 | 32266.8 | 8693.2 |
| CLIP reference 峰值 | 40 blocks + CLIP | 33024.8 | — | 34555 | 6405 |
| CLIP offload 后 | 40 blocks | 31811.7 | — | 32352.8 | 8607.2 |
| 81 帧 DiT 峰值 | 40 blocks | 首步可见 37592.7 | 约 38574 | **40081** | **879** |
| Segment cleanup 后 | 40 blocks | 31828.9 | — | 约 32389 | 约 8571 |
| 卸载 7 blocks 后 | 33 blocks | 26437.1 | 26510.0 | 27028.8 | 13931.2 |
| VAE decode 峰值 | 33 blocks | 35399.9 | 37952.0 | **38470.8** | **2489.2** |
| DiT 恢复后 | 40 blocks | 31819.7 | — | 32368.8 | 8591.2 |

完整任务峰值仍由 DiT 决定，而不是 VAE。VAE 阶段卸载 7 个 block 后已有约 2.43 GiB 余量；DiT 阶段只有约 0.86 GiB 余量，是当前单卡方案真正的显存边界。

CPU 侧同时保留约 31272 MiB 的 DiT master。若以后每张 GPU 启动一个独立 worker，应把每个 worker 的这部分 host memory、page cache 和 conditioning 占用一起计入整机容量规划。

## 6. 实验递进与结果

| 实验 | 日志 | 结果 |
|---|---|---|
| 初始化 | `experiment_logs/single_gpu/101-20260828-180700.log` | CPU/GPU 各保留完整 DiT；加载 12.138 秒，正常退出 |
| 首个 diffusion step | `experiment_logs/single_gpu/101-20260828-180805.log` | 40 个 block 全部通过；NVML 峰值 39973 MiB |
| 首 segment 全 6 步 | `experiment_logs/single_gpu/101-20260828-181018.log` | 无 DiT offload，6/6 步成功；耗时 100.2 秒，峰值 40073 MiB |
| 单 segment + VAE 切换 | `experiment_logs/single_gpu/101-20260828-182908.log` | offload、VAE、reload 和输出全部成功；峰值 40033 MiB |
| 完整 4 segments | `experiment_logs/single_gpu/101-20260828-183620.log` | 24 步、4 次 VAE、3 次 history encode、4 次 reload 全部成功 |
| 在线 CLIP 首步 | `experiment_logs/single_gpu/101-20260828-191658.log` | CLIP context 与缓存完全一致；offload 后首步 latent hash 不变 |
| 在线 CLIP 完整回归 | `experiment_logs/single_gpu/101-20260828-191836.log` | 297 帧和 driving audio 全部成功；视频码流与原单卡输出一致 |

完整回归证明阶段切换不是只能执行一次：第二至第四个 segment 均在 reload 后继续完成 DiT，CPU master 和恢复后的 CUDA DiT 大小也保持一致。

## 7. 单卡与双卡 FSDP 的效率对比

双卡对照日志为 `experiment_logs/fsdp_baseline/101-20260828-152421.log`。

| 指标 | 单卡 | 双卡 FSDP | 差异 |
|---|---:|---:|---:|
| 81 帧 DiT | 16.70 秒/步 | 16.91--16.99 秒/步 | 单卡略快 |
| 57 帧 DiT | 10.31 秒/步 | 10.49--10.53 秒/步 | 单卡略快 |
| 在线 CLIP | 1.331 秒 | 使用缓存 | 单卡新增 |
| DiT block offload | 合计 0.586 秒 | 不适用 | 单卡新增 |
| DiT block reload | 合计 5.465 秒 | 不适用 | 单卡新增 |
| Audio mux | 2.599 秒 | 2.431 秒 | 单卡慢 0.168 秒 |
| 当前请求总时间 | 429.555 秒 | 424.714 秒 | 单卡慢 4.841 秒（1.14%） |
| 使用 GPU 数 | 1 | 2 | 单卡节省 1 张 GPU |

此前不执行在线 CLIP且关闭音频的历史单卡回归为 425.044 秒，不能直接与含音频的双卡结果比较；它的生成主流程比双卡慢约 0.77%。

当前正式 `--full-inference` 使用 driving audio，其他 probe 仍不输出音频。带在线 CLIP 和 driving audio 的完整回归为 429.555 秒；双卡 FSDP 含音频对照为 424.714 秒，单卡慢 4.841 秒（1.14%）。其中在线 CLIP 热态编码/offload 为 1.331 秒，音频 mux 为 2.599 秒。

## 8. 输出结果与数值差异

完整单卡输出为：

```text
experiment_outputs/single_gpu/101-20260828-183620.mp4
```

视频为 H.264、512×896、297 帧、9.9 秒。单卡输出 SHA-256 为：

```text
fa56145b030db5f6659be2449ff68fe67850344f52801c97762a677c19c71e70
```

双卡 FSDP 输出 SHA-256 为：

```text
7039c5f231eb64b544c4aa288ea5107411c9e7f51bdcf4c93d125d6e1610680a
```

两个 MP4 不能直接用文件哈希判断模型一致性，因为历史单卡输出没有音频，而双卡输出包含 AAC。对解码后 RGB 视频逐帧比较，average PSNR 为 30.640 dB，SSIM 为 0.935252，说明视频接近但并非逐像素相同。

差异在任何 VAE 阶段 offload 发生前就已经出现：首个 diffusion step 的单卡与双卡 latent 均值只差约 `9.90e-6`，但 tensor hash 不同。现有证据更支持普通参数布局与 FSDP flatten/all-gather 路径导致矩阵 kernel、对齐或 BF16 累加顺序不同，而不是 VAE 阶段权重切换直接造成差异。

目前尚未做“相同单卡执行拓扑，仅切换 offload 开/关”的中间 latent 对照，因此还不能单独量化阶段切换是否在后续 segment 引入额外数值差异。

在线 CLIP 完整回归输出为 `experiment_outputs/single_gpu/101-20260828-191836.mp4`。它包含 H.264 297 帧和 AAC 9.9 秒音频；视频 elementary stream 与上述历史单卡输出完全一致，音频 stream 与双卡 driving audio 完全一致。因此重新执行 CLIP 没有改变单卡视频结果。

## 9. 使用方式

完整单卡推理：

```bash
python run_single_gpu_experiment.py --full-inference
```

入口默认使用物理 GPU 2、固定 conditioning cache、FFN/RoPE 8192 分块，并把日志保存到 `experiment_logs/single_gpu/`，输出保存到 `experiment_outputs/single_gpu/`。

可用的递进验证模式：

```bash
python run_single_gpu_experiment.py --init-only
python run_single_gpu_experiment.py --memory-probe
python run_single_gpu_experiment.py --dit-segment-probe
python run_single_gpu_experiment.py --vae-offload-probe
```

## 10. 当前限制与下一步

1. DiT 阶段只剩约 879 MiB 物理余量，新增 CUDA 常驻对象前必须重新测峰值。
2. 需要做同一单卡拓扑下 offload 开/关的 latent hash 对照，隔离权重阶段切换本身的数值影响。
3. 在线 CLIP 已验证可行，但使每个 worker 的 CPU RSS 增加约 1.2 GiB，并增加 checkpoint 加载时间；多 worker 时需要计入主存和启动 I/O。
4. 当前缓存仍包含仅用于校验的 `clip_context`；稳定后可将缓存 schema 拆成纯 T5 context，避免语义冗余。
5. 若目标扩展为多单卡 worker，应同时评估每个 worker 约 31.27 GiB 的 CPU master 和模型文件 I/O 对主存及启动并发的影响。

## 11. 相关文件

- 单卡入口：`run_single_gpu_experiment.py`
- 推理配置与调用：`scail2_inference/engine.py`
- DiT/VAE 阶段切换：`wan/scail.py`
- 完整实验流水：`SCAIL2_EXPERIMENT_CONTEXT.md`
- DiT 显存细节：`SCAIL2_DIT_MEMORY_OPTIMIZATION_REPORT.md`
- 双卡完整显存：`SCAIL2_FULL_MEMORY_PROFILE_REPORT.md`
- VAE buffer 剖析：`SCAIL2_VAE_MEMORY_PROFILE_REPORT.md`
