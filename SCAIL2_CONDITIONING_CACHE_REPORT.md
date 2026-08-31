# SCAIL2 T5 / CLIP Conditioning Cache 报告

日期：2026-08-27

> 2026-08-31 更新：本文主体记录的是早期 T5/CLIP 混合缓存实验，现已被
> `scail2-t5-cache-v1` 取代。当前运行时接口只接收 `t5_cache_path`，文件中
> 仅含 `text_context` 和 `negative_context`；CLIP 根据每次请求的 reference
> image 在线计算。当前实现与验证结果见 `SCAIL2_EXPERIMENT_CONTEXT.md` 第 39 节。

## 1. 目标

将 T5 文本编码和 CLIP 参考图编码从 DiT 主推理 worker 中移出，提前生成可复用的 conditioning cache。主推理只加载 VAE、DiT 和缓存 tensor，不再构造或加载 T5、CLIP 模型。

该改动主要用于：

- 减少主 worker 的模型加载时间。
- 释放 T5、CLIP 在主 worker 中占用的显存和内存。
- 为后续单卡 DiT 推理实验创造显存空间。
- 对固定 prompt 或重复参考图复用预处理结果。

## 2. 实现方式

独立预处理入口为：

```text
prepare_conditioning.py
```

预处理在单张 GPU 上顺序执行：

1. 加载 T5，计算正向和负向 prompt context。
2. 释放 T5。
3. 加载 CLIP，计算参考图 visual context。
4. 释放 CLIP，将三个 tensor 保存为 safetensors 文件。

当前缓存位置：

```text
/raid/scail2_exp/experiment_cache/conditioning/101.safetensors
```

`experiment_cache/` 已加入 `.gitignore`。

## 3. 缓存内容

| Tensor | Shape | dtype | 大小 |
|---|---:|---:|---:|
| `text_context` | `[92, 4096]` | BF16 | 753,664 bytes |
| `negative_context` | `[1, 4096]` | BF16 | 8,192 bytes |
| `clip_context` | `[1, 257, 1280]` | FP16 | 657,920 bytes |
| safetensors 文件头及元数据 | — | — | 1,056 bytes |
| **合计** | — | — | **1,420,832 bytes（1.355 MiB）** |

缓存保存的是完整 T5 encoder output 和 CLIP visual transformer output，而不是 token ID。因此主推理不需要重新加载或执行 T5、CLIP。

## 4. 缓存校验

缓存通过以下信息绑定到原始输入和模型版本：

- Prompt SHA-256。
- Negative prompt SHA-256。
- 参考图路径和 SHA-256。
- 目标宽高。
- T5、CLIP checkpoint 路径、文件大小和 mtime。
- Cache schema 版本。
- Tensor 名称、shape、dtype 和 device。

任意身份字段或 tensor 契约不匹配时，主推理会拒绝使用缓存，不会静默使用过期结果。

## 5. 主推理行为

使用缓存时需要显式传入：

```bash
python -u run_fsdp_experiment.py \
  --physical-gpus 2,3 \
  --conditioning-cache experiment_cache/conditioning/101.safetensors \
  --expandable-segments
```

此时两个 rank 均执行以下路径：

- `precomputed_conditioning=True`。
- `t5_fsdp=False`。
- `self.text_encoder=None`，不构造或加载 T5。
- `self.clip=None`，不构造或加载 CLIP。
- 只读取约 1.42 MB 的 cache，并将三个 tensor 移动到当前 GPU。

上述行为仍是默认路径。单卡实验后续增加了可选
`online_clip_conditioning=True`：继续读取缓存中的 T5 context，但重新执行
CLIP reference encode，并在生成 visual context 后把 CLIP offload 回 CPU。缓存中的
`clip_context` 当前只用于结果一致性校验；双卡路径不启用该开关。

代码仍会导入 T5/CLIP Python 模块，并读取 checkpoint 的 stat 信息做身份校验，但不会读取 checkpoint tensor payload。

缓存不会仅因文件存在而自动启用。如果不传 `--conditioning-cache`，主流程仍会加载 T5 和 CLIP。

## 6. 实验结果

### 预处理

日志：`experiment_logs/conditioning/101-20260827-172426.log`

| 阶段 | 耗时 | GPU peak allocated |
|---|---:|---:|
| T5 加载及两次编码 | 3.598 s | 11088.6 MiB |
| CLIP 加载及编码 | 2.170 s | 2491.2 MiB |
| 预处理进程总计 | 12.5 s | 分阶段执行 |

### 主流程

| 指标 | 在线 T5/CLIP | 使用 cache | 变化 |
|---|---:|---:|---:|
| `engine_load` | 17.862 s | 13.771 s | -4.091 s |
| ready 到任务完成 | 437.707 s | 436.904 s | -0.803 s |

完整缓存推理日志：`experiment_logs/fsdp_baseline/101-20260827-172549.log`。

四个 segment、每段六个 sampling step 全部完成，进程 exit code 0，没有 OOM、NCCL timeout、traceback 或 AMP `FutureWarning`。

缓存前后输出 MP4 的 SHA-256 完全相同：

```text
7039c5f231eb64b544c4aa288ea5107411c9e7f51bdcf4c93d125d6e1610680a
```

这确认固定输入、prompt 和 seed 下，独立预处理没有改变最终输出。

## 7. 结论与后续方向

预处理只使用一次时不会缩短整体耗时，因为 12.5 秒的预处理成本大于主流程节省的约 4.9 秒。该方案的主要收益是模型解耦、缓存复用和主 worker 显存释放，而不是加速 DiT sampling。

当前固定 prompt 可以全局复用 T5 context。若未来存在大量任务，建议进一步拆分：

- T5 cache 按 prompt hash 单独存储，当前约 744 KiB。
- CLIP cache 按参考图 hash 单独存储，当前约 642.5 KiB。
- 主任务只组合对应的 text cache 和 image cache，避免重复保存固定 prompt context。

下一阶段可以在不加载 T5/CLIP 的前提下测试单卡 DiT + VAE 的真实显存峰值和完整推理可行性。
