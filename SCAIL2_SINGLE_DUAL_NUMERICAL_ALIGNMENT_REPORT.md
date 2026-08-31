# SCAIL2 单卡与双卡结果对齐报告

日期：2026-08-31

## 1. 结论

单卡和双卡 FSDP 结果不一致的原因已经确定：双卡根 FSDP wrapper 会在 DiT forward 前把浮点输入统一转换为 BF16，而普通单卡模型没有这层隐式转换。缓存或在线 CLIP 生成的 visual context 原本是 FP16，因此两条路径从 `clip_embedding` 开始产生不同结果。

实验曾让单卡显式复现 FSDP 的根输入转换规则；转换后，63 个中间检查点、首步 latent、最终 H.264/AAC 码流和完整 MP4 均与旧双卡逐字节一致。这证明入口 dtype 是差异的充分原因。

最终不采用该转换作为标准。新的数值标准是保留 CLIP context 原始 FP16，不在 DiT 入口预先转换为 BF16。代码分析和 FP32 参考实验均表明，后续算子不会因为这次 BF16 转换而提高计算精度；相反，转换会先丢失 FP16 尾数精度。后续应让双卡路径对齐不转换的单卡语义。

## 2. 问题背景

早期完整单卡推理能够正常生成视频，但结果与双卡基准不同：

| 项目 | 旧单卡 | 双卡 FSDP |
|---|---|---|
| 首步 latent mean | 0.001903103 | 0.001893205 |
| 首步 latent SHA-256 | `26517d3b...9764eb3` | `f6845ee3...85261` |

旧单卡与双卡解码视频的对比结果为 PSNR 30.640 dB、SSIM 0.935252，说明两者视觉上接近，但不是同一个数值结果。

差异在第一次 VAE decode 和 DiT block offload 之前已经出现，因此可以排除视频封装、音频、VAE 阶段切换和后续 segment history 处理是最早原因。

## 3. 定位方法

对比实验固定以下条件：

- 使用同一个 `experiment_cache/conditioning/101.safetensors`。
- 不加载或执行 T5/CLIP，避免在线预处理波动。
- seed 固定为 42。
- FFN 和 RoPE 分块均为 8192。
- 只运行第一个 segment 的第一个 diffusion step。

在单卡和双卡路径中，对 DiT 输入、embedding、block 0 的 modulation、self-attention、cross-attention、FFN 和最终 head 共记录 63 个 tensor 的 shape、dtype、统计值和 SHA-256。

初始对照日志：

- 单卡：`experiment_logs/single_gpu/101-20260831-112917.log`
- 双卡：`experiment_logs/fsdp_baseline/101-20260831-113234.log`

这种逐层对比避免了仅根据最终视频反推原因，也排除了“FSDP 参数 flatten 或不同 GEMM kernel 一定造成差异”的早期猜测。

## 4. 最早差异与根因

双卡配置使用 FSDP `MixedPrecision`。其根 wrapper 默认启用 `cast_root_forward_inputs=True`，因此进入 DiT forward 的所有浮点 tensor 都先转换为参数 dtype BF16。

普通单卡模型没有 FSDP wrapper：

- `clip_context` 保持 CLIP 输出/缓存中的 FP16。
- 部分其他输入保持 FP32。
- 后续算子虽然处于 BF16 autocast，但并不等价于在模型入口统一转换。

逐层 trace 显示：

1. 单卡和双卡的模型权重、噪声及主要输入内容一致。
2. visual context 的入口 dtype 分别为 FP16 和 BF16。
3. 第一个不同的计算结果出现在 `model.clip_embedding`。
4. 差异随后进入 cross-attention image K/V 并向后传播。
5. 在该点之前，block 0 self-attention 和 text K/V 完全一致。

因此，入口 dtype 差异是单卡和双卡输出不一致的充分原因，而不是 FSDP 参数布局、collective 或 VAE offload。

## 5. FP16 与 BF16 的精度分析

另使用实际 conditioning、`img_emb` 和 block 0 image K/V 权重建立 FP32 参考，日志为：

```text
experiment_logs/numerical/clip-context-dtype-20260831.log
```

实际 CLIP context 的绝对值最大为 10.65625，远低于 FP16 范围，不存在通过 BF16 更大指数范围避免溢出的需求。实际计算路径为：

```text
FP16/BF16 context
  -> LayerNorm（内部/输出 FP32）
  -> Linear（BF16 输出）
  -> GELU（BF16）
  -> Linear（BF16 输出）
  -> LayerNorm（FP32 输出）
  -> image K/V Linear（BF16 输出）
  -> FlashAttention（BF16）
```

该路径没有因为输入为 BF16 而获得更高的算术精度。相对 FP32 参考，原始 FP16 输入在各层的 RMSE/MAE 反而更小。因此不能把 FP16 到 BF16 的入口转换描述为“精度优化”；它会进行一次不可逆舍入，其价值是复现双卡 FSDP 语义。

## 6. 最终标准选择

最终采用以下规则：

- 以不做入口 dtype 转换的单卡路径作为新的数值标准。
- CLIP visual context 保持其原始 FP16 dtype 进入 `clip_embedding`，不预先转换为 BF16。
- 不修改 CLIP 输出，也不修改 conditioning cache 的存储 dtype。
- 双卡 FSDP 路径需要避免根 wrapper 对 CLIP context 的隐式 BF16 转换，并以新的单卡标准重新完成中间 tensor 和最终输出回归。
- `cast_dit_forward_inputs` 只保留为复现旧双卡结果和诊断差异的实验能力，不再作为目标执行方式。

选择不转换不是因为 FP16 可以让后续 BF16 GEMM 变成 FP16 计算，而是因为第一个 LayerNorm 会保留输入信息并以 FP32 处理；入口先转 BF16只会提前量化数据，无法被后续 FP32/BF16 算子恢复。

## 7. BF16 对齐实验：用于确认根因

下面的结果记录了“单卡增加入口 BF16 转换”的对照实验。它用于证明根因，但不再代表最终采用的数值标准。

对齐后的首步日志为：

```text
experiment_logs/single_gpu/101-20260831-113743.log
```

自动对比结果为：

| 指标 | 结果 |
|---|---:|
| 单卡 trace stages | 63 |
| 双卡 trace stages | 63 |
| 共同 stages | 63 |
| 不一致 stages | 0 |

对齐后的首步 latent：

```text
mean=0.001893205
std=0.961710393
SHA-256=f6845ee32b24e01bb80b8f6dfa3467c62119bb3014ef94f65718a40bd8085261
```

BF16 对齐方式的完整单卡回归：

- 日志：`experiment_logs/single_gpu/101-20260831-124017.log`
- 输出：`experiment_outputs/single_gpu/101-20260831-124017.mp4`
- 双卡基准：`experiment_outputs/fsdp_baseline/101-20260828-152421.mp4`

| 验证对象 | 单卡与双卡结果 |
|---|---|
| 在线 CLIP context 与缓存 | 逐元素一致 |
| 帧数/分辨率/帧率 | 297 / 512×896 / 30 fps，一致 |
| MP4 大小 | 19587416 bytes，一致 |
| H.264 stream SHA-256 | `67357588...bf36b17`，一致 |
| AAC stream SHA-256 | `b2645001...40f046`，一致 |
| MP4 SHA-256 | `7039c5f2...10680a`，一致 |

该实验说明：只要单卡复现 FSDP 的入口 BF16 转换，旧双卡结果就可以逐字节重现，因此已排除 FSDP 参数 flatten、collective 或 VAE 阶段切换是结果差异的必要原因。但“能够复现旧双卡”不等于“该数值规则更合理”。

### 7.1 新标准的单卡完整回归

单卡入口已经改为默认保留输入 dtype；原 BF16 转换改为仅由 `--cast-dit-forward-inputs` 显式开启的诊断能力。本次新标准回归同时使用缓存 CLIP context：

- 日志：`experiment_logs/single_gpu/101-20260831-140335.log`
- 输出：`experiment_outputs/single_gpu/101-20260831-140335.mp4`
- 历史不转换对照：`experiment_outputs/single_gpu/101-20260828-191836.mp4`

任务正常完成 297 帧和 driving audio。新输出与历史不转换版本均为 19652849 bytes，MP4 SHA-256 均为：

```text
fcb0871b57305440b8cd33ab8e3960a7d8f81f68c401190addda80ac59137d7e
```

两者的 H.264 和 AAC stream 也分别逐字节一致。这既建立了不转换路径的新单卡标准，也再次确认缓存 CLIP context 与在线编码结果相同。

## 8. 最终结论

单卡与双卡差异已经从“可能的执行拓扑数值误差”收敛为一个明确且可控制的输入 dtype 行为。BF16 对齐实验完成了根因验证，但最终标准改为不转换 CLIP context，保留其原始 FP16 精度。

因此后续工作的方向不是继续让单卡复现旧双卡，而是修改双卡的入口转换行为，使其对齐不转换的单卡路径。单卡入口和完整回归已经按新标准完成；双卡完成代码调整后，仍需要重新执行 63 个中间检查点对比、首步 latent 对比和完整 MP4 回归，再建立新的双卡基准。
