# SCAIL2 DiT 推理显存分布与优化报告

日期：2026-08-28

## 1. 结论

当前 81 帧、512×896、双卡 FSDP 推理中，DiT 显存峰值主要来自长序列 self-attention，而不是 T5/CLIP conditioning。

本轮最终验证六项无损优化：

1. FFN 按 8192 token 分块。
2. RoPE 保留 FP64/complex128 内部计算，但输出直接恢复为 BF16。
3. cross-attention 输出合并和 block residual 使用原地累加。
4. RoPE 保留 FP64/complex128 计算，并按 8192 token 分块执行。
5. 完整长度的 FlashAttention K/V 使用 view，跳过冗余 `cat` 复制。
6. Cross-attention 在 Q projection 后立即释放完整长度的归一化 query 输入。

在 block 0 的细粒度测量中：

- 最高 allocated peak 从 26350.1 MiB 降至 23193.4 MiB，减少 3156.7 MiB。
- 两个 rank 的 allocator 最大 reserved 从 27500 MiB 降至 23740 MiB，减少 3760 MiB。
- FFN 计算局部峰值减少 1785.3 MiB。
- cross-attention 后半段的 live allocation 最多减少约 953.8 MiB。
- 最新完整推理耗时为 424.714 秒，相比 436.904 秒减少 12.190 秒（2.79%）；这是单次运行结果。
- 完整输出 MP4 与优化前逐字节一致。

RoPE 内部 FP32/complex64 和 BF16 block residual 虽然还能降低部分占用，但会改变推理结果，因此未作为默认优化采用。

## 2. 测量条件和口径

固定条件：

- GPU：物理 GPU 2、3，NVIDIA A100-SXM4-40GB。
- 模型：SCAIL-14B BF16。
- 输入：`testdata/101`。
- 分辨率：512×896。
- segment：81 pixel frames，对应 latent `T=21, H=112, W=64`。
- 双卡 FSDP `FULL_SHARD`。
- 使用预计算 conditioning cache，不加载 T5 和 CLIP。
- allocator：`expandable_segments:True`。
- 诊断只执行第一个 segment、一个 diffusion step，不执行 VAE decode 和视频写出。
- 细粒度日志只记录 block 0。

诊断字段含义：

| 字段 | 含义 |
|---|---|
| `allocated_mib` | 当前阶段结束时仍存活的 PyTorch CUDA tensor 大小 |
| `phase_peak_mib` | 从该阶段开始到结束期间的最大 allocated |
| `phase_increase_mib` | 阶段峰值相对阶段入口 allocated 的增量 |
| `reserved_mib` | PyTorch CUDA allocator 已向驱动保留的总空间，不等于存活 tensor |

每个阶段开始前会调用 `reset_peak_memory_stats()`，因此 `phase_peak_mib` 可以定位该阶段的瞬时峰值。埋点不执行额外的 device-wide synchronize。

block 0 第一个阶段开始前的 allocated 约为：

```text
21104.8 - 1907.9 = 19196.9 MiB
```

这约 18.75 GiB 包括 DiT FSDP shard、当前 block 参数、VAE、输入 latent、embedding 和运行时 buffer。它不是纯激活显存。

注意：block 0 的初始 hidden 为 BF16，完成第一个 residual 后变为 FP32。后续 block 因而会比 block 0 入口多保留约 476.9 MiB。本文的绝对峰值是 block 0 的实测值；各阶段张量大小和优化差值仍可代表后续同形状 block，但不能把 block 0 峰值直接当成整个 diffusion step 的全局峰值。

## 3. 序列和主要张量大小

当前 self-attention 序列共 48832 token：

| 序列部分 | Token 数 | 占比 |
|---|---:|---:|
| Reference | 1792 | 3.67% |
| Video | 37632 | 77.06% |
| Pose | 9408 | 19.27% |
| 合计 | 48832 | 100% |

模型 hidden dim 为 5120，FFN dim 为 13824，共 40 个 block。主要张量的理论大小如下：

| 对象 | Shape / 说明 | 大小 |
|---|---|---:|
| 完整 DiT 参数 | BF16 checkpoint | 30.539 GiB |
| 每 rank 理想参数 shard | 完整参数的 1/2 | 约 15.270 GiB |
| 单个完整 block 参数 | FSDP forward 时 all-gather | 770.3 MiB |
| 主 hidden，BF16 | `[1,48832,5120]` | 476.9 MiB |
| 主 hidden，FP32 | `[1,48832,5120]` | 953.8 MiB |
| Q/K/V，BF16 | 三份 `[1,48832,5120]` | 1430.6 MiB |
| 原始 RoPE Q/K 输出 | 两份 FP32 | 1907.5 MiB |
| 优化后 RoPE Q/K 输出 | 两份 BF16 | 953.8 MiB |
| 完整 FFN 中间结果 | `[1,48832,13824]` BF16 | 1287.6 MiB |
| 8192-token FFN 中间结果 | `[1,8192,13824]` BF16 | 216.0 MiB |
| T5+CLIP 投影后 context | `[1,769,5120]` BF16 | 7.5 MiB |

conditioning context 只有约 7.5 MiB。真正的大对象是 self-attention 的长 query 序列、Q/K/V、RoPE 临时量、FP32 residual 和 FFN 中间结果。

FlashAttention 已经避免创建完整的 `48832×48832` attention matrix，因此当前不存在 LLM 式的巨大 attention score matrix 或长期 KV cache。

## 4. 未优化 block 0 的显存分布

基线日志：`experiment_logs/fsdp_baseline/101-20260828-130305.log`

### 4.1 阶段入口的基础占用 B

第一个埋点开始前的实测 allocated 为 19196.9 MiB，以下记作基础占用 `B`。它在后面的主表中不再逐行重复。

| B 中的数据 | 大小 | 说明 |
|---|---:|---|
| DiT 参数 shard | 约 15636.1 MiB | 30.539 GiB BF16 checkpoint 理想二等分；实际 FSDP flat parameter 存在少量 padding/管理差异 |
| 当前完整 block 参数 | 约 770.3 MiB | 当前 block forward 所需的完整 BF16 参数/all-gather buffer |
| block 0 初始 hidden | 476.9 MiB | `[1,48832,5120]` BF16 |
| VAE、DiT 非 block 参数、输入 latent、embedding、FSDP/CUDA 运行时等 | 约 2313.7 MiB | 用 `19196.9 - 15636.1 - 770.3 - 476.9` 得到的剩余项，无法仅由 allocator 统计继续可靠拆分 |
| **基础占用 B 合计** | **19196.9 MiB** | 实测/推导基准 |

上表中的参数 shard 和 block buffer 是按 checkpoint/FSDP 结构估算；`B` 的总量 19196.9 MiB 是实测值。后续阶段的长序列 tensor 可以根据 shape 和 dtype 精确计算，并且与 allocated 差值逐项吻合。

### 4.2 各阶段数据占用大表

“B 之外存活数据”表示该阶段日志输出时仍然存活的 tensor；“峰值时额外临时数据”是 `phase_peak_mib - allocated_mib`，表示运算过程中短暂存在、到阶段结束已经释放的数据。多个 tensor 的生命期会重叠，因此不能把不同行简单相加。下表所有显存数值的单位均为 MiB。

| 模块 | 阶段 | 存活数据 1 | 存活数据 2 | 存活数据 3 | 存活数据 4 | 存活数据 5 | 存活数据 6 | 存活数据 7 | B 外存活小计 | 峰值时额外临时数据 | 实测阶段结束 allocated | 实测阶段峰值 |
|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|
| Self-attention | Norm + modulation 输入 | Self 输入 FP32<br>953.8 | — | — | — | — | — | — | 953.8 | LayerNorm/modulation 中间结果约 954.1 | 20150.7 | 21104.8 |
| Self-attention | Q/K/V projection | Self 输入 FP32<br>953.8 | Q/K/V BF16<br>1430.6 | — | — | — | — | — | 2384.4 | autocast projection、RMSNorm 旧/新 tensor 重叠约 1430.8 | 21581.3 | 23012.1 |
| Self-attention | Q RoPE | Self 输入 FP32<br>953.8 | Q/K/V BF16<br>1430.6 | Q RoPE FP32<br>953.8 | — | — | — | — | 3338.2 | Q 的 FP64/complex128 转换、乘法、stack/cast 临时量约 2793.0 | 22535.1 | 25328.1 |
| Self-attention | K RoPE | Self 输入 FP32<br>953.8 | Q/K/V BF16<br>1430.6 | Q RoPE FP32<br>953.8 | K RoPE FP32<br>953.8 | — | — | — | 4291.9 | K 的 FP64/complex128 临时量约 2793.0 | 23488.8 | 26281.8 |
| Self-attention | FlashAttention | Self 输入 FP32<br>953.8 | Q/K/V BF16<br>1430.6 | Q RoPE FP32<br>953.8 | K RoPE FP32<br>953.8 | Attention 输出 FP32<br>953.8 | — | — | 5245.7 | Q/K BF16 cast、FlashAttention BF16 输出及 workspace 重叠约 1907.5 | 24442.6 | **26350.1** |
| Self-attention | Output projection | Self 输入 FP32<br>953.8 | Q/K/V BF16<br>1430.6 | Projected y BF16<br>476.9 | — | — | — | — | 2861.3 | FP32 attention 输出、autocast BF16 输入/输出和 GEMM 临时量约 1431.6 | 22058.2 | 23489.8 |
| Self-attention | Self residual | Projected y BF16<br>476.9 | 新 hidden FP32<br>953.8 | — | — | — | — | — | 1430.7 | `y*e` 和 residual 新旧结果重叠约 953.7 | 20627.6 | 21581.3 |
| Cross-attention | Cross Q/K/V | Self y BF16<br>476.9 | 当前 hidden FP32<br>953.8 | Query 输入 FP32<br>953.8 | Cross Q BF16<br>476.9 | Context K/V BF16<br>约 15.0 | — | — | 2876.3 | Norm/RMSNorm、projection autocast 旧/新 tensor 约 1892.7 | 22073.2 | 23965.9 |
| Cross-attention | CLIP image attention | Self y BF16<br>476.9 | 当前 hidden FP32<br>953.8 | Query 输入 FP32<br>953.8 | Cross Q BF16<br>476.9 | Context K/V BF16<br>约 15.0 | Image 输出 BF16<br>476.9 | — | 3353.2 | FlashAttention workspace 约 7.4 | 22550.1 | 22557.5 |
| Cross-attention | T5 text attention | Self y BF16<br>476.9 | 当前 hidden FP32<br>953.8 | Query 输入 FP32<br>953.8 | Cross Q BF16<br>476.9 | Context K/V BF16<br>约 15.0 | Image 输出 BF16<br>476.9 | Text 输出 BF16<br>476.9 | 3830.1 | FlashAttention workspace 约 7.4 | 23027.0 | 23034.4 |
| Cross-attention | Image/text 输出合并 | Self y BF16<br>476.9 | 当前 hidden FP32<br>953.8 | Query 输入 FP32<br>953.8 | Cross Q BF16<br>476.9 | Context K/V BF16<br>约 15.0 | Image 输出 BF16<br>476.9 | 合并输出 BF16<br>476.9 | 3830.1 | `text_output + image_output` 新分配 476.8 | 23027.0 | 23503.8 |
| Cross-attention | Output projection | Self y BF16<br>476.9 | 当前 hidden FP32<br>953.8 | Query 输入 FP32<br>953.8 | Cross Q BF16<br>476.9 | Context K/V BF16<br>约 15.0 | Image 输出 BF16<br>476.9 | Projected 输出 BF16<br>476.9 | 3830.1 | autocast/GEMM 输出重叠约 477.8 | 23027.0 | 23504.8 |
| Cross-attention | Cross residual 整体 | Self y BF16<br>476.9 | 旧 hidden FP32<br>953.8 | 新 hidden FP32<br>953.8 | — | — | — | — | 2384.4 | 该埋点包围整个 cross 子图，cross 输出和新 residual 的最大重叠约 1923.5 | 21581.3 | 23504.8 |
| FFN | Norm + modulation 输入 | Self y BF16<br>476.9 | 调用者 hidden FP32<br>953.8 | 当前 hidden FP32<br>953.8 | FFN 输入 FP32<br>953.8 | — | — | — | 3338.2 | LayerNorm/modulation 中间结果约 953.7 | 22535.1 | 23488.8 |
| FFN | 第一层 Linear | Self y BF16<br>476.9 | 调用者 hidden FP32<br>953.8 | 当前 hidden FP32<br>953.8 | FFN 输入 FP32<br>953.8 | FFN hidden BF16<br>1287.6 | — | — | 4625.7 | autocast 输入和 GEMM 临时量约 477.9 | 23822.6 | **24300.5** |
| FFN | GELU | Self y BF16<br>476.9 | 调用者 hidden FP32<br>953.8 | 当前 hidden FP32<br>953.8 | GELU 输出 BF16<br>1287.6 | — | — | — | 3672.0 | GELU 输入/输出同时存在约 1287.5 | 22868.9 | 24156.4 |
| FFN | 第二层 Linear | Self y BF16<br>476.9 | 调用者 hidden FP32<br>953.8 | 当前 hidden FP32<br>953.8 | FFN hidden BF16<br>1287.6 | FFN 输出 BF16<br>476.9 | — | — | 4148.8 | GEMM workspace 约 1.0 | 23345.7 | 23346.7 |
| FFN | FFN residual | Self y BF16<br>476.9 | 调用者 hidden FP32<br>953.8 | FFN y BF16<br>476.9 | 新 hidden FP32<br>953.8 | — | — | — | 2861.3 | `y*e` 和 FP32 residual 新旧结果重叠约 1907.5 | 22058.2 | 23965.7 |

这张表显示：

- Self-attention 的最高点由 Q/K/V、两份 FP32 RoPE 输出、FP32 attention 输出和 FP64/complex128 RoPE 临时量重叠形成。
- Cross-attention 的 context K/V 只有约 15 MiB；主要占用仍然是 48832-token query 和多份 476.9/953.8 MiB 输出。
- FFN 第一层的 1287.6 MiB 中间 tensor 与此前 OOM 中 `Tried to allocate 1.26 GiB` 完全对应。
- Python 调用层级会让调用者的 FP32 hidden 在子函数返回前保持存活，因此一些阶段会同时看到两份 953.8 MiB FP32 hidden；原地复用能够提前结束这种重叠。
- 仅分块 Linear 不足以消除 FFN 峰值，因为完整 FP32 norm/modulation 输入本身也是 953.8 MiB；因此实际优化将 norm、modulation 和两层 FFN 一起分块。

## 5. 优化后 block 0 的显存分布

最终优化诊断日志：`experiment_logs/fsdp_baseline/101-20260828-152305.log`

基础占用仍使用第 4.1 节的 `B = 19196.9 MiB`。下表的结构和口径与未优化表一致，所有显存数值的单位均为 MiB。FFN 和 RoPE 均按 8192 token 分块；FFN 埋点覆盖整个分块循环，因此原来的五个 FFN 阶段在这里合并为“8192-token 分块计算”一行。

| 模块 | 阶段 | 存活数据 1 | 存活数据 2 | 存活数据 3 | 存活数据 4 | 存活数据 5 | 存活数据 6 | 存活数据 7 | B 外存活小计 | 峰值时额外临时数据 | 实测阶段结束 allocated | 实测阶段峰值 |
|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|
| Self-attention | Norm + modulation 输入 | Self 输入 FP32<br>953.8 | — | — | — | — | — | — | 953.8 | LayerNorm/modulation 中间结果约 954.1 | 20150.7 | 21104.8 |
| Self-attention | Q/K/V projection | Self 输入 FP32<br>953.8 | Q/K/V BF16<br>1430.6 | — | — | — | — | — | 2384.4 | autocast projection、RMSNorm 旧/新 tensor 重叠约 1430.8 | 21581.3 | 23012.1 |
| Self-attention | Q RoPE | Self 输入 FP32<br>953.8 | Q/K/V BF16<br>1430.6 | Q RoPE BF16<br>476.9 | — | — | — | — | 2861.3 | 分块 FP64/complex128 临时量约 658.4 | 22058.2 | 22716.6 |
| Self-attention | K RoPE | Self 输入 FP32<br>953.8 | Q/K/V BF16<br>1430.6 | Q RoPE BF16<br>476.9 | K RoPE BF16<br>476.9 | — | — | — | 3338.2 | 分块 FP64/complex128 临时量约 658.3 | 22535.1 | 23193.4 |
| Self-attention | FlashAttention | Self 输入 FP32<br>953.8 | Q/K/V BF16<br>1430.6 | Q RoPE BF16<br>476.9 | K RoPE BF16<br>476.9 | Attention 输出 BF16<br>476.9 | — | — | 3815.0 | FlashAttention workspace 约 7.5 | 23011.9 | 23019.4 |
| Self-attention | Output projection | Self 输入 FP32<br>953.8 | Q/K/V BF16<br>1430.6 | Projected y BF16<br>476.9 | — | — | — | — | 2861.3 | autocast/GEMM 输出重叠约 477.9 | 22058.2 | 22536.1 |
| Self-attention | Self residual | Projected y BF16<br>476.9 | 新 hidden FP32<br>953.8 | — | — | — | — | — | 1430.7 | `y*e` 和 residual 新旧结果重叠约 953.7 | 20627.6 | 21581.3 |
| Cross-attention | Cross Q/K/V | Self y BF16<br>476.9 | 当前 hidden FP32<br>953.8 | Cross Q BF16<br>476.9 | Context K/V BF16<br>约 15.0 | — | — | — | 1922.6 | Query norm/RMSNorm、projection 临时量约 1892.6 | 21119.5 | 23012.1 |
| Cross-attention | CLIP image attention | Self y BF16<br>476.9 | 当前 hidden FP32<br>953.8 | Cross Q BF16<br>476.9 | Context K/V BF16<br>约 15.0 | Image 输出 BF16<br>476.9 | — | — | 2399.5 | FlashAttention workspace 约 7.5 | 21596.3 | 21603.8 |
| Cross-attention | T5 text attention | Self y BF16<br>476.9 | 当前 hidden FP32<br>953.8 | Cross Q BF16<br>476.9 | Context K/V BF16<br>约 15.0 | Image 输出 BF16<br>476.9 | Text 输出 BF16<br>476.9 | — | 2876.4 | FlashAttention workspace 约 7.5 | 22073.2 | 22080.7 |
| Cross-attention | Image/text 输出合并 | Self y BF16<br>476.9 | 当前 hidden FP32<br>953.8 | Cross Q BF16<br>476.9 | Context K/V BF16<br>约 15.0 | Image 输出 BF16<br>476.9 | 合并输出 BF16<br>476.9 | — | 2876.4 | 原地合并，无新增 tensor<br>0.0 | 22073.2 | 22073.2 |
| Cross-attention | Output projection | Self y BF16<br>476.9 | 当前 hidden FP32<br>953.8 | Cross Q BF16<br>476.9 | Context K/V BF16<br>约 15.0 | Projected 输出 BF16<br>476.9 | — | — | 2399.5 | autocast/GEMM 输出重叠约 477.9 | 21596.3 | 22074.2 |
| Cross-attention | Cross residual 整体 | Self y BF16<br>476.9 | 原地 hidden FP32<br>953.8 | — | — | — | — | — | 1430.7 | cross 子图最大重叠约 1446.6 | 20627.6 | 22074.2 |
| FFN | 8192-token 分块计算 | Self y BF16<br>476.9 | 当前 hidden FP32<br>953.8 | FFN 输出 BF16<br>476.9 | — | — | — | — | 1907.5 | 单 chunk 输入、hidden 和 GEMM 临时量约 457.0 | 21104.4 | 21561.4 |
| FFN | FFN residual | Self y BF16<br>476.9 | 旧 hidden FP32<br>953.8 | FFN y BF16<br>476.9 | 新 hidden FP32<br>953.8 | — | — | — | 2861.3 | `y*e` 和 residual 新旧结果重叠约 953.7 | 22058.2 | 23011.9 |

与未优化表直接对照可见：RoPE 的两份长期输出各从 953.8 降到 476.9，FP64/complex128 临时量又通过分块从约 2885 降到约 658；完整长度 K/V 不再通过 `cat` 复制 953.8；cross query 的 953.8 MiB FP32 归一化输入在 Q projection 后立即释放；cross merge 和 cross residual 不再创建新的长序列 tensor；FFN 完整的 1287.6 中间结果被 8192-token chunk 的短生命周期临时量替代。

## 6. 各项优化的显存差异

### 6.1 FFN 8192-token 分块

对照日志：

- 未分块：`101-20260828-130305.log`
- 分块：`101-20260828-130414.log`

实现方式：

- 每次处理最多 8192 token。
- 每个 chunk 独立执行 norm、modulation、Linear、GELU、Linear。
- 预先分配完整 BF16 输出 tensor，逐 chunk 写入，避免最终 `cat` 再复制一份输出。

| 指标 | 未分块 | 8192-token 分块 | 变化 |
|---|---:|---:|---:|
| FFN 计算最高绝对峰值 | 24300.5 MiB | 22515.2 MiB | **-1785.3 MiB** |
| FFN 计算阶段峰值增量 | 最高 1907.5 MiB | 933.9 MiB | 约 -973.6 MiB |
| block 0 最高峰 | 26350.1 MiB | 26350.1 MiB | 0 |

分块显著降低 FFN 峰值，但当时整个 block 的最高峰仍由 self-attention 决定，因此 block 0 的最高值没有变化。

首步 latent SHA-256 与基线完全一致，说明当前 GPU/kernel 路径下分块没有引入数值差异。

### 6.2 RoPE 输出直接恢复为 BF16

对照日志：

- FFN 分块、RoPE FP32 输出：`101-20260828-130414.log`
- FFN 分块、RoPE BF16 输出：`101-20260828-130545.log`

RoPE 仍使用 FP64/complex128 做内部旋转，只把最终输出从固定 `.float()` 改为输入 Q/K 的 BF16 dtype。FlashAttention 原本就会立即将 Q/K 转为 BF16，因此该修改只是提前执行已有的精度边界，并避免 FlashAttention 输出被冗余转成 FP32。

| 阶段 | 修改前峰值 | 修改后峰值 | 变化 |
|---|---:|---:|---:|
| Q RoPE | 25328.1 MiB | 24943.1 MiB | -385.0 MiB |
| K RoPE | 26281.8 MiB | 25419.9 MiB | -861.9 MiB |
| FlashAttention | 26350.1 MiB | 23973.1 MiB | **-2377.0 MiB** |
| Self output projection | 23489.8 MiB | 22536.1 MiB | -953.7 MiB |
| block 0 最高峰 | 26350.1 MiB | 25419.9 MiB | **-930.2 MiB** |
| 最大 reserved | 27500 MiB | 25740 MiB | **-1760 MiB** |

FlashAttention 阶段降幅大于最终 block 峰值降幅，是因为修改后最高点从 FlashAttention 转移到了 K RoPE。最终 block 峰值由仍使用 FP64/complex128 的 K RoPE 决定。

首步 latent SHA-256 与基线完全一致。

### 6.3 FlashAttention 完整长度 K/V fast path

对照日志：

- 修改前：`101-20260828-142607.log`
- 修改后：`101-20260828-150011.log`

当所有 `k_lens` 都等于物理 K/V 长度时，直接使用 `flatten(0, 1)` view；存在 padding 或变长序列时仍回退到原来的切片和 `cat`。当前 self-attention 的 `k_lens=[48832]`，因此可以避免两份 476.9 MiB 的完整 K/V 副本。

| 指标 | 修改前 | Fast path | 变化 |
|---|---:|---:|---:|
| FlashAttention 峰值 | 23973.1 MiB | 23019.4 MiB | **-953.7 MiB** |
| 阶段峰值增量 | 1438.1 MiB | 484.3 MiB | -953.8 MiB |
| 峰值高于阶段结束 allocated | 961.2 MiB | 7.5 MiB | -953.7 MiB |
| block 0 最高峰 | 23973.1 MiB | 23965.9 MiB | -7.2 MiB |
| rank 0 最大 reserved | 24700 MiB | 24700 MiB | 0 |

该优化没有改变原始 Q/K 的释放时机；它们仍存活到 self-attention forward 返回。首步 latent SHA-256 与基线一致，完整 MP4 也逐字节一致。全局最高点转移到 Cross Q/K/V。

### 6.4 RoPE 8192-token 分块

对照日志：

- 未分块：`101-20260828-130841.log`
- 8192-token 分块：`101-20260828-142607.log`

RoPE 继续使用 FP64 输入和 complex128 乘法，只把完整序列改为分块转换、旋转并写入预分配的 BF16 输出。不存在跨 token 的归约运算。

| 指标 | 未分块 | 8192-token 分块 | 变化 |
|---|---:|---:|---:|
| Q RoPE 峰值 | 24943.1 MiB | 22716.6 MiB | -2226.5 MiB |
| K RoPE 峰值 | 25419.9 MiB | 23193.4 MiB | -2226.5 MiB |
| RoPE 阶段峰值增量 | 3361.8 MiB | 1135.2 MiB | -2226.6 MiB |
| block 0 最高峰 | 25419.9 MiB | 23973.1 MiB | **-1446.8 MiB** |
| 最大 reserved | 25740 MiB | 24700 MiB | **-1040 MiB** |

最高点已经从 K RoPE 转移到 FlashAttention；Cross Q/K/V 的 23965.9 MiB 仅低 7.2 MiB。首步 latent SHA-256 和完整 MP4 SHA-256 均与未分块版本完全一致。

### 6.5 Cross-attention 原地复用

对照日志：

- 修改前：`101-20260828-130545.log`
- 修改后：`101-20260828-130841.log`

修改包括：

- text attention 输出直接原地加上 image attention 输出。
- cross-attention 输出直接原地加到已有 FP32 block hidden。

| 阶段 | 修改前峰值 | 修改后峰值 | 变化 |
|---|---:|---:|---:|
| Image/text 输出合并 | 23503.8 MiB | 23027.0 MiB | -476.8 MiB |
| Cross output projection | 23504.8 MiB | 23028.0 MiB | -476.8 MiB |
| Cross residual 整体 | 23504.8 MiB | 23028.0 MiB | -476.8 MiB |
| 后续 FFN chunk | 22515.2 MiB | 21561.4 MiB | -953.8 MiB |
| 后续 FFN residual | 23965.7 MiB | 23011.9 MiB | -953.8 MiB |
| block 0 最高峰 | 25419.9 MiB | 25419.9 MiB | 0 |

该修改消除了 cross merge 的 476.9 MiB 新 tensor，并让旧 FP32 residual 更早释放，因此后续阶段的 live allocation 最多下降约 953.8 MiB。全局峰值没有变化，因为 K RoPE 仍然更高。

首步 latent SHA-256 与基线完全一致。

### 6.6 Cross-attention query 输入提前释放

对照日志：

- 修改前：`101-20260828-150011.log`
- 修改后：`101-20260828-152305.log`

原实现先在调用方计算 `self.norm3(x)`，再把完整的 FP32 query 输入作为参数传入 cross-attention。Python 调用参数会让这份 `[1,48832,5120]`、953.8 MiB 的 tensor 一直存活到整个 cross-attention 返回。

修改后把原始 hidden 和 `norm3` 传入 cross-attention；内部仍以完整长度依次执行相同的 LayerNorm、Q Linear 和 RMSNorm，但在 Q Linear 返回后立即删除归一化输入。该方案没有分块，也没有改变 GEMM 形状或运算顺序。

| 指标 | 修改前 | 提前释放后 | 变化 |
|---|---:|---:|---:|
| Cross Q/K/V 阶段结束 allocated | 22073.2 MiB | 21119.5 MiB | **-953.7 MiB** |
| Cross Q/K/V 峰值 | 23965.9 MiB | 23012.1 MiB | **-953.8 MiB** |
| Cross image attention 峰值 | 22557.5 MiB | 21603.8 MiB | -953.7 MiB |
| Cross text attention 峰值 | 23034.4 MiB | 22080.7 MiB | -953.7 MiB |
| Cross output projection 峰值 | 23028.0 MiB | 22074.2 MiB | -953.8 MiB |
| block 0 最高峰 | 23965.9 MiB | 23193.4 MiB | **-772.5 MiB** |
| rank 0 最大 reserved | 24700 MiB | 23740 MiB | **-960 MiB** |
| rank 1 最大 reserved | 25180 MiB | 23740 MiB | **-1440 MiB** |

Cross Q/K/V 不再是最高点；block 0 的最高点转移到 K RoPE 的 23193.4 MiB。首步 latent SHA-256 和完整 MP4 SHA-256 均与修改前完全一致。

## 7. 无损组合前后的阶段对比

最终无损组合为：

- FFN chunk size 8192。
- RoPE 输出为 BF16，内部 FP64/complex128 计算按 8192 token 分块。
- cross-attention 原地复用。
- 完整长度 FlashAttention K/V 使用 view fast path。
- Cross-attention query 输入在 Q projection 后立即释放。
- FP32 block residual 保持不变。

| 阶段 | 原始峰值 | 无损组合峰值 | 变化 |
|---|---:|---:|---:|
| Self-attention 输入 | 21104.8 MiB | 21104.8 MiB | 0 |
| Self Q/K/V | 23012.1 MiB | 23012.1 MiB | 0 |
| Q RoPE | 25328.1 MiB | 22716.6 MiB | -2611.5 MiB |
| K RoPE | 26281.8 MiB | 23193.4 MiB | -3088.4 MiB |
| FlashAttention | 26350.1 MiB | 23019.4 MiB | -3330.7 MiB |
| Self output projection | 23489.8 MiB | 22536.1 MiB | -953.7 MiB |
| Cross Q/K/V | 23965.9 MiB | 23012.1 MiB | -953.8 MiB |
| Cross merge | 23503.8 MiB | 22073.2 MiB | -1430.6 MiB |
| Cross residual | 23504.8 MiB | 22074.2 MiB | -1430.6 MiB |
| FFN 计算 | 24300.5 MiB | 21561.4 MiB | **-2739.1 MiB** |
| FFN residual | 23965.7 MiB | 23011.9 MiB | -953.8 MiB |
| block 0 最高峰 | 26350.1 MiB | 23193.4 MiB | **-3156.7 MiB** |
| rank 0 最大 reserved | 27500 MiB | 23740 MiB | **-3760 MiB** |

`FFN 计算` 的累计差值同时包含 FFN 分块和 cross residual 更早释放带来的收益，因此不能把表中各行差值直接相加。

## 8. 未采用的低精度方案

### 8.1 RoPE 内部 FP32/complex64

日志：`experiment_logs/fsdp_baseline/101-20260828-130707.log`

| 指标 | FP64 内部计算 | FP32 内部计算 | 变化 |
|---|---:|---:|---:|
| K RoPE 峰值 | 25419.9 MiB | 23927.6 MiB | -1492.3 MiB |
| block 0 最高峰 | 25419.9 MiB | 23969.1 MiB | -1450.8 MiB |
| 相对原始基线 | 26350.1 MiB | 23969.1 MiB | -2381.0 MiB |

但最终 latent SHA-256 从：

```text
f6845ee32b24e01bb80b8f6dfa3467c62119bb3014ef94f65718a40bd8085261
```

变为：

```text
a261e85bc9b4f02b9d9bc55f0e42f5b2823d6de30522c6cb3d79d467f562305f
```

latent mean 从 `0.001893205` 变为 `0.001888150`。说明 FP32 频率和旋转误差会跨过后续 BF16 舍入边界，因此该修改已经回退。

### 8.2 BF16 block residual

日志：`experiment_logs/fsdp_baseline/101-20260828-131010.log`

| 阶段 | FP32 residual allocated | BF16 residual allocated | 变化 |
|---|---:|---:|---:|
| Self residual 结束 | 20627.6 MiB | 20150.7 MiB | -476.9 MiB |
| Cross residual 结束 | 20627.6 MiB | 20150.7 MiB | -476.9 MiB |
| FFN chunk 结束 | 21104.4 MiB | 20627.6 MiB | -476.8 MiB |
| block 0 最高峰 | 25419.9 MiB | 25419.9 MiB | 0 |

BF16 residual 能降低跨阶段常驻 hidden，但 block 0 的最高峰仍由 K RoPE 决定。最终 latent SHA-256 变为：

```text
9f79e0b38d54fbf3e5748a176a59a3ee5922a63e6238b9c6ae61d64a001ee826
```

因此默认继续使用 FP32 residual。代码只保留 `--bf16-residual` 作为后续质量实验开关。

## 9. 完整推理回归

最终无损组合命令：

```bash
python run_fsdp_experiment.py \
  --conditioning-cache experiment_cache/conditioning/101.safetensors \
  --physical-gpus 2,3 \
  --ffn-chunk-size 8192 \
  --rope-chunk-size 8192 \
  --expandable-segments
```

日志和输出：

- 日志：`experiment_logs/fsdp_baseline/101-20260828-152421.log`
- 输出：`experiment_outputs/fsdp_baseline/101-20260828-152421.mp4`

| 指标 | Conditioning cache 基线 | RoPE 分块优化 | K/V fast path | Cross query 释放 |
|---|---:|---:|---:|---:|
| Ready 到完成 | 436.904 秒 | 426.297 秒 | 424.583 秒 | 424.714 秒 |
| 相对基线 | — | -10.607 秒（-2.43%） | -12.321 秒（-2.82%） | -12.190 秒（-2.79%） |
| 输出帧数 | 297 | 297 | 297 | 297 |
| 输出大小 | 19587416 bytes | 19587416 bytes | 19587416 bytes | 19587416 bytes |
| SHA-256 | `7039c5...0680a` | `7039c5...0680a` | `7039c5...0680a` | `7039c5...0680a` |

完整 SHA-256：

```text
7039c5f231eb64b544c4aa288ea5107411c9e7f51bdcf4c93d125d6e1610680a
```

完整 4 segment、每段 6 step 均成功，没有 OOM、NCCL timeout 或 traceback。

## 10. 当前代码状态

- `--memory-probe`：诊断模式，默认关闭。
- `--ffn-chunk-size 8192`：启用已验证的 FFN 分块；默认 0，便于继续 A/B。
- `--rope-chunk-size 8192`：启用已验证的 FP64/complex128 RoPE 分块；默认 0，便于继续 A/B。
- RoPE BF16 输出：已作为无损代码优化保留。
- Cross-attention 原地复用：已作为无损代码优化保留。
- 完整长度 FlashAttention K/V fast path：已保留；变长或含 padding 时自动回退到 `cat`。
- Cross-attention query 输入提前释放：已保留；仍以完整长度执行 LayerNorm、Q Linear 和 RMSNorm。
- RoPE FP32/complex64 内部计算：已回退。
- `--bf16-residual`：默认关闭，仅供实验。

当前最高峰已变为 K RoPE 的 23193.4 MiB；Cross Q/K/V 为 23012.1 MiB，Self FlashAttention 为 23019.4 MiB。三个峰值已经非常接近，下一步若要继续明显降低 block 峰值，需要同时考虑 RoPE 分块粒度、Self Q/K/V 生命周期和 Cross Q RMSNorm 临时量，单独降低其中一项可能只会再次转移最高点。
