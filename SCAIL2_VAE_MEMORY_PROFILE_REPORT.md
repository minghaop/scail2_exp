# SCAIL2 VAE Decode 各阶段 Buffer 显存报告

日期：2026-08-28

## 1. 实验范围与结论

本报告只分析 **rank 0 / 物理 GPU 2 上第一个 81 帧 segment 的 VAE decode**。输入 latent 为 FP32 `[1,16,21,112,64]`，输出为 FP32 `[1,3,81,896,512]`。VAE 参数和所有 decode activation 均为 FP32。

数据来源：

- 日志：`experiment_logs/fsdp_baseline/101-20260828-173206.log`
- 输出：`experiment_outputs/fsdp_baseline/101-20260828-173206.mp4`
- 模式：实验时临时使用 `--vae-memory-profile`；数据采集完成后，该入口和 VAE 内部 profiling 埋点已从运行时代码移除
- 每个 causal conv 分别在 cache clone、temporal cat、padding 和 convolution 后同步采样，并对每个操作单独重置 allocator peak。

最重要的结论：

1. decode 开始前的常驻 allocated 为 `16654.5 MiB`；VAE decode 的 live allocated 峰值为 `25596.3 MiB`，新增 `8941.8 MiB`。
2. 32 个稳定 causal cache 共 `4137.875 MiB`，其中 896×512 cache 占 `2352 MiB`。
3. live 峰值发生在 896×512 residual causal conv，不在最终视频输出拼接。
4. 峰值时可以用已知 buffer 和卷积临时 workspace 完整还原到 `25596.3 MiB`。
5. 1008 MiB temporal cat 只在 `F.pad` 执行时与 1014.196 MiB padded tensor 重叠；`F.pad` 返回后 cat 已释放，因此单独融合 cat+pad 不能消除后续 causal-conv 的全局最高峰。

除特别注明外，下表数值单位均为 MiB。

## 2. Decode 开始前的常驻显存

| Buffer | 大小 | 说明 |
|---|---:|---|
| DiT BF16 参数分片 | 15636.010 | 16395544384 个 BF16 参数由两个 rank 各持有一半；理论 tensor payload |
| VAE FP32 参数 | 484.057 | 194 个 tensor，126892531 个参数；两个 rank 各复制一份 |
| 其他 PyTorch 常驻 | 约 534.434 | conditioning、final latent、FSDP/模型对象的其他 tensor 等 |
| **decode 开始 allocated** | **16654.5** | PyTorch 当前存活 tensor 总量 |
| allocator reserved | 16798.0 | 包括 allocated 和 allocator 空闲块 |
| device used | 17922.8 | reserved 再加 CUDA/NCCL/driver 等非 PyTorch 分配 |

这里的 DiT 分片和 VAE 参数在整个 decode 过程中始终常驻。VAE 激活并不是在一张空卡上运行，而是叠加在约 16.65 GiB 的 PyTorch resident set 上。

## 3. Latent 预处理阶段

| 阶段 | 新产生的 buffer | 大小 | 阶段结束 allocated | 操作峰值 | 生命周期 |
|---|---|---:|---:|---:|---|
| decode 输入 | caller final latent `[1,16,21,112,64]` | 9.188 | 16654.5 | — | caller 在 decode 外仍持有，因此包含在开始基线中 |
| scale | scaled latent `[1,16,21,112,64]` | 9.188 | 16663.7 | 16672.9 | 保留到 decode 返回 |
| `conv2` | transformed latent `x` `[1,16,21,112,64]` | 9.188 | 16672.9 | 16682.0 | 保留全部 21 个时间步，逐步切 view 输入 decoder |
| 单步 latent view | `[1,16,1,112,64]` | 0 | — | — | view，不新增 storage；对应 payload 为 0.438 |

完成 `conv2` 后，相对 decode 开始稳定多出两份 9.188 MiB tensor：scaled latent 和 transformed latent，合计 `18.375 MiB`。原始 final latent 已计入 16654.5 MiB 基线。

## 4. Decoder 每个稳定时间步的特征流

第 0 个 latent 时间步只输出 1 帧；第 1--20 步各输出 4 帧。下表是第 20 步的稳定形状。

| 阶段 | 主 feature | 大小 | 输出帧数 | 说明 |
|---|---|---:|---:|---|
| latent slice | `[1,16,1,112,64]` | 0.438 | — | `x[:, :, i:i+1]` view |
| decoder conv1/middle | `[1,384,1,112,64]` | 10.500 | 1 latent frame | 低分辨率 middle 含 residual 和 attention |
| 第一次时空上采样后 | `[1,192,2,224,128]` | 42.000 | 2 feature frames | temporal ×2、spatial ×2、channel 384→192 |
| 224×128 residual stage | `[1,384,2,224,128]` | 84.000 | 2 | 第一块 residual 将 channel 192→384 |
| 第二次时空上采样后 | `[1,192,4,448,256]` | 336.000 | 4 | temporal ×2、spatial ×2、channel 384→192 |
| 448×256 residual stage | `[1,192,4,448,256]` | 336.000 | 4 | 三个 residual block |
| 最后一次空间上采样后 | `[1,96,4,896,512]` | 672.000 | 4 | 仅 spatial ×2、channel 192→96 |
| 896×512 residual/head | `[1,96,4,896,512]` | 672.000 | 4 | 三个 residual block、norm、SiLU |
| RGB chunk | `[1,3,4,896,512]` | 21.000 | 4 | head causal conv 输出 |

## 5. 持久 Causal Cache 分布

每个 causal conv 保存前两个时间帧，供下一个 latent 时间步使用。第 0 步只建立单帧 cache，共 `2016.438 MiB`；从第 1 步开始达到稳定的双帧 cache，共 `4137.875 MiB`。

| 分辨率 | Cache buffer | 数量 | 单项/小计 | 合计 |
|---|---|---:|---:|---:|
| 112×64 | latent cache `[1,16,2,112,64]` | 1 | 0.875 | 0.875 |
| 112×64 | conv cache `[1,384,2,112,64]` | 11 | 21.000 | 231.000 |
| **112×64 小计** |  | **12** |  | **231.875** |
| 224×128 | temporal-upsample cache | 1 | 42.000 | 42.000 |
| 224×128 | conv cache `[1,384,2,224,128]` | 6 | 84.000 | 504.000 |
| **224×128 小计** |  | **7** |  | **546.000** |
| 448×256 | conv cache `[1,192,2,448,256]` | 6 | 168.000 | **1008.000** |
| 896×512 | conv cache `[1,96,2,896,512]` | 7 | 336.000 | **2352.000** |
| **总计** |  | **32** |  | **4137.875** |

cache 在一个 decoder 时间步内还有“旧 cache + 新 cache clone”的短暂重叠。旧 cache 属于上表的 4137.875 MiB；新 clone 在当前卷积完成后替换旧 cache。最高分辨率每次 clone 额外产生 336 MiB。

## 6. 四档分辨率 Causal Conv 的逐 Buffer 分解

下表取第 20 个 latent 时间步中每档分辨率 operation peak 最高的 causal conv。此时共有三类跨层持久 buffer：

- 非 decoder 时步 buffer：约 `16672.9`，即 decode 开始基线加 scaled latent 和 transformed latent；
- 旧 causal cache：`4137.875`；
- 已累计的 77 帧 RGB 输出：`404.250`。

三者相加为该时间步开始时的 `21215.0 MiB`。表内“input/shortcut”等 buffer 均叠加在这 21215.0 MiB 上。

| 分辨率 | Input / shortcut | Norm activation | 新 cache clone | Temporal cat | Padded input | Conv output | Workspace/其他临时量 | Operation peak |
|---|---|---|---|---|---|---|---|---:|
| 112×64 | alias<br>10.500 | 10.500 | 21.000 | 31.500<br>pad 后释放 | 33.064 | 10.500 | 约 48.300 | 21348.8 |
| 224×128 | input 42.000<br>projected shortcut 84.000 | 42.000 | 42.000 | 84.000<br>pad 后释放 | 86.074 | 84.000 | 约 597.700 | 22192.8 |
| 448×256 | alias<br>336.000 | 336.000 | 168.000 | 504.000<br>pad 后释放 | 510.205 | 336.000 | 约 514.000 | 23415.2 |
| 896×512 | alias<br>672.000 | 672.000 | 336.000 | 1008.000<br>pad 后释放 | 1014.196 | 672.000 | 约 1015.100 | **25596.3** |

说明：

- `alias` 表示 residual shortcut 是 `Identity`，input 与 shortcut 指向同一 storage，不重复计数。
- workspace/其他临时量由 `operation_peak_allocated - conv_end allocated` 推算；其他 tensor 大小均由实测 shape、dtype 和元素数精确计算。
- temporal cat 和 padded input 只在 padding kernel 执行期间重叠；上表将二者都列出是为了展示阶段 buffer，不能在 causal-conv 峰值分解中再次同时相加。

## 7. 最高分辨率 Residual Causal Conv 的完整时间线

第 20 步、`decoder.upsamples.12.residual.2` 的逐操作数据如下：

| 时点 | 当前新增/存活 buffer | 当前 allocated | 本操作峰值 | 已释放内容 |
|---|---|---:|---:|---|
| timestep 开始 | 稳定 cache 4137.875；77 帧输出 404.250；scaled/x latent 18.375 | 21215.0 | — | — |
| residual 入口 | feature/shortcut 672 | 21887.0 | 21887.0 | 前一层 feature 已替换 |
| norm 后 | norm activation 672 | 22559.0 | 23903.0 | normalize 内部约 1344 临时量已释放 |
| cache clone 后 | new cache 336 | 22895.0 | 22895.0 | — |
| temporal cat 后 | cat `[1,96,6,896,512]` 1008 | 23903.0 | 23903.0 | — |
| padding 执行中 | cat 1008 + padded 1014.196 短暂共存 | — | 24917.2 | — |
| padding 返回 | padded 1014.196 | 23909.2 | — | **cat 1008 已释放** |
| convolution 返回 | padded 1014.196 + output 672 | 24581.2 | **25596.3** | 约 1015.1 workspace 已释放 |
| residual block 返回 | feature 672 + residual 中间状态 | 23231.0 | — | padded input 和 conv local 已释放 |
| decoder layer 返回 | feature 672 | 21887.0 | — | residual local/shortcut 引用已释放 |

这证明当前代码的 `x = F.pad(x, padding)` 已经会在返回后释放 temporal cat；显式 `del` 不会进一步降低后续 convolution 峰值。

## 8. 最高峰的完整 Buffer 等式

896×512 causal conv 的 `25596.3 MiB` live allocated 可按同时存活的 buffer 还原：

| 同时存活的内容 | 大小 |
|---|---:|
| decode 开始常驻集合 | 16654.500 |
| scaled latent | 9.188 |
| transformed latent `x` | 9.188 |
| 旧 causal cache | 4137.875 |
| 已累计 77 帧输出 | 404.250 |
| 当前 feature / identity shortcut | 672.000 |
| norm/SiLU 后 activation | 672.000 |
| 新 causal cache clone | 336.000 |
| padded conv input | 1014.196 |
| conv output | 672.000 |
| cuDNN workspace/其他 conv 临时量（推算） | 1015.100 |
| **合计** | **25596.296** |
| **allocator 实测 operation peak** | **25596.3** |

两者在日志精度内完全一致。相对 decode 开始的新增量为：

```text
25596.3 - 16654.5 = 8941.8 MiB
```

## 9. 空间上采样阶段的 Buffer

| 上采样 | 输入 feature | Temporal 输出 | Nearest 输出 | Conv2d 输出 | Conv2d workspace/其他临时量 | 阶段 operation peak |
|---|---:|---:|---:|---:|---:|---:|
| 112×64 → 224×128 | 10.500 | 21.000 | 84.000 | 42.000 | 约 86.5 | 21438.0 |
| 224×128 → 448×256 | 84.000 | 168.000 | 672.000 | 336.000 | 约 674.5 | 22981.5 |
| 448×256 → 896×512 | 336.000 | 无 temporal upsample | 1344.000 | 672.000 | 约 1344.6 | **24911.6** |

最后一级空间上采样的 Conv2d 是仅次于 causal conv 的大峰值之一。Nearest 输出 1344 MiB、Conv2d 输出 672 MiB 和约 1344.6 MiB workspace 在操作内重叠。

## 10. 输出累计 Buffer

第 0 步输出 1 帧 5.25 MiB；之后每步增加 4 帧 21 MiB。cache 在第 1 步达到稳定大小，后续每步的峰值增加量正好约为新增输出的 21 MiB。

| Latent 时间步 | 累计 RGB 帧 | 累计输出 | Causal cache | 步结束 allocated | 步内最高 allocated |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 5.250 | 2016.438 | 18694.5 | 20040.4 |
| 1 | 5 | 26.250 | 4137.875 | 20837.0 | 24861.3 |
| 2 | 9 | 47.250 | 4137.875 | 20858.0 | 25218.3 |
| 3 | 13 | 68.250 | 4137.875 | 20879.0 | 25239.3 |
| 4 | 17 | 89.250 | 4137.875 | 20900.0 | 25260.3 |
| 5 | 21 | 110.250 | 4137.875 | 20921.0 | 25281.3 |
| 6 | 25 | 131.250 | 4137.875 | 20942.0 | 25302.3 |
| 7 | 29 | 152.250 | 4137.875 | 20963.0 | 25323.3 |
| 8 | 33 | 173.250 | 4137.875 | 20984.0 | 25344.3 |
| 9 | 37 | 194.250 | 4137.875 | 21005.0 | 25365.3 |
| 10 | 41 | 215.250 | 4137.875 | 21026.0 | 25386.3 |
| 11 | 45 | 236.250 | 4137.875 | 21047.0 | 25407.3 |
| 12 | 49 | 257.250 | 4137.875 | 21068.0 | 25428.3 |
| 13 | 53 | 278.250 | 4137.875 | 21089.0 | 25449.3 |
| 14 | 57 | 299.250 | 4137.875 | 21110.0 | 25470.3 |
| 15 | 61 | 320.250 | 4137.875 | 21131.0 | 25491.3 |
| 16 | 65 | 341.250 | 4137.875 | 21152.0 | 25512.3 |
| 17 | 69 | 362.250 | 4137.875 | 21173.0 | 25533.3 |
| 18 | 73 | 383.250 | 4137.875 | 21194.0 | 25554.3 |
| 19 | 77 | 404.250 | 4137.875 | 21215.0 | 25575.3 |
| 20 | 81 | 425.250 | 4137.875 | 21236.0 | **25596.3** |

最终一次输出拼接时同时存在：

| 输出 buffer | 大小 |
|---|---:|
| 原 77 帧输出 | 404.250 |
| 当前 4 帧 chunk | 21.000 |
| 新 81 帧输出 | 425.250 |
| **输出相关瞬时合计** | **850.500** |

但输出 cat 的 allocator operation peak 只有 `21661.2 MiB`，远低于 causal conv 的 `25596.3 MiB`。输出累计不是当前第一优先级。

## 11. Decode 结束与 allocator/device 口径

| 时点 | allocated | reserved | device used |
|---|---:|---:|---:|
| timestep 20 结束 | 21236.0 | 28660.0 | 29784.8 |
| clear causal cache 后 | 17098.1 | 28660.0 | 29784.8 |
| 释放的 causal cache | **4137.9** | 0 | 0 |

`21236.0 - 17098.1 = 4137.9 MiB`，与 cache 精确统计 `4137.875 MiB` 一致。allocated 立即下降，但 reserved/device used 不下降，是因为 caching allocator 保留了已释放块。外部 NVML 因此只能看到约 29.8 GiB 平台，不能判断每个 buffer 是否仍存活。

本轮父进程 NVML 峰值：

| GPU | 峰值 | 阶段 |
|---|---:|---|
| 物理 GPU 2 / rank 0 | **29785** | VAE decode |
| 物理 GPU 3 / rank 1 | **27467** | DiT；不执行 decode |

## 12. 优化优先级

按当前实测，优先级建议为：

1. **896×512 causal conv**：处理 1014.196 MiB padded input 和约 1015.1 MiB workspace；这是 live 全局峰值的直接决定者。
2. **高分辨率 causal cache**：896×512 cache 为 2352 MiB，全部 cache 合计 4137.875 MiB；需要评估更小 dtype、按层重算或改变流式 cache 策略对结果的影响。
3. **最后一级空间 Conv2d**：1344 MiB nearest 输出与约 1344.6 MiB workspace 重叠，operation peak 为 24911.6 MiB。
4. **Temporal cat + pad 融合**：可以消除 padding 阶段约 1008 MiB cat/padded 重叠，但 cat 在 convolution 前已经释放，所以只做这一项不会降低 25596.3 MiB 的最高 causal-conv 峰值。
5. **输出累计**：最终输出相关瞬时 buffer 为 850.5 MiB，但绝对峰值较低，优先级低于 causal cache、padding 和卷积 workspace。

所有可能改变卷积 padding、dtype、分块或 cache 内容的优化，都需要用完整 297 帧推理进行 MP4 SHA-256 一致性验证。

## 13. Profiling 代码状态

本报告的数据和原始日志保留，但临时使用的 `--vae-memory-profile`、单 segment 特殊执行路径、VAE 内同步显存埋点及相关环境开关已经从运行时代码移除。当前普通推理路径没有这些同步点或逐操作日志开销。

如果未来需要复测，应基于本报告列出的采样点重新加入最小化临时埋点；同步埋点会显著增加日志量和 VAE 执行时间，因此不能用来衡量 decode 性能。
