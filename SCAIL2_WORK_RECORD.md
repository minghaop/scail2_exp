# SCAIL2 工作记录

## DiT 显存基线分析

首先对 DiT 首个 diffusion step 增加分阶段显存统计，用于确认主要显存来源，并建立结果一致性基线。

分析确认显存压力主要来自长序列 attention、RoPE 和 FFN，而不是 conditioning。实验同时记录 latent hash，作为后续优化是否改变推理结果的判断依据。

## DiT 显存优化

依次完成 FFN 分块、RoPE 输出优化、cross-attention 临时量优化、RoPE 分块、FlashAttention 完整长度 fast path，以及 Cross Query 生命周期缩短。

这些修改的目的是减少长序列计算中的大块临时 tensor、重复拷贝和不必要的保活。

优化后，完整双卡推理耗时从 436.904 秒降至 424.714 秒，减少 12.190 秒，约 2.79%。输出 MP4 与优化前逐字节一致。

降低 RoPE 内部精度和使用 BF16 block residual 虽然还能减少部分显存，但会改变输出结果，因此没有作为默认方案使用。

## 完整流程显存分析

在完成 DiT 局部优化后，对完整 4 个 segment、24 个 diffusion step、VAE decode 和输出流程进行了设备级显存测量。

这项工作的目的是解释外部 GPU 监控约 27G 与此前 block 级报告约 23G 之间的差异。

结果确认，约 23G 只代表特定 DiT block 的 PyTorch live tensor；完整 DiT 的设备实际占用约为 27.5G，其中还包括 allocator 缓存和 CUDA/驱动占用。完整双卡任务的最高点实际出现在 GPU 2 的 VAE decode，约为 29.7G。

## VAE 显存排查

随后对单个 VAE decode 的各阶段 buffer 进行了详细测量，用于判断约 9 GiB 新增显存的具体来源，以及是否存在可以直接释放的大对象。

排查确认峰值主要来自最高分辨率 causal convolution、长期 causal cache 和卷积 workspace，而不是最终输出拼接。padding 前的旧 buffer 也已经按预期释放，额外增加显式释放不会降低后续卷积峰值。

考虑到优化 VAE 需要改动卷积或 cache 策略，成本和结果风险较高，因此暂时不修改 VAE。数据采集完成后，临时 profiling 代码已移除，保留实验日志和报告。

## 单卡 DiT 可行性验证

在双卡显存优化基础上，建立了不依赖 `torchrun`、process group 和 FSDP 的独立单卡入口。

验证顺序为初始化、首个 diffusion step、首个 segment 全 6 步。实验确认完整 40-block DiT 可以在一张 40GB A100 上执行，DiT 运行期间不需要 CPU offload。

单卡 DiT 峰值约为 40073 MiB，只剩约 887 MiB 物理余量。因此 VAE、CLIP 或其他较大 CUDA buffer 不能与完整 DiT 同时驻留。

## VAE 阶段 DiT 权重切换

为了在不改变 DiT 执行方式的前提下运行 VAE，在 diffusion 完成后临时卸载末尾 7 个 DiT block，VAE 和 history encode 结束后再从 CPU master 恢复。

7 个 block 共释放 5391.8 MiB。卸载后 VAE 峰值为 38470.8 MiB，距离 40GB 上限约有 2489.2 MiB 余量。恢复后的 DiT 能够继续执行下一段推理。

## 完整单卡回归

最后完成了 297 帧完整单卡推理。4 个 segment、24 个 diffusion step、4 次 VAE decode、3 次 history encode 和 4 次 DiT 恢复全部成功。

单卡总耗时为 425.044 秒，双卡 FSDP 对照为 424.714 秒。两者单请求延迟基本相同，但单卡方案只占用一张 GPU。使用相同的两张 GPU 启动两个独立单卡 worker 时，理论并发吞吐接近原双卡 worker 的两倍。

本次历史单卡输出没有音频，正式单卡模式随后已恢复 driving audio。单卡与双卡解码后视频的平均 PSNR 为 30.640 dB，SSIM 为 0.935252。首个 diffusion step 在 VAE 权重切换前已经出现微小数值差异，因此差异主要与普通单卡参数路径和 FSDP 数值路径不同有关。

## 相关记录

- `SCAIL2_DIT_MEMORY_OPTIMIZATION_REPORT.md`
- `SCAIL2_FULL_MEMORY_PROFILE_REPORT.md`
- `SCAIL2_VAE_MEMORY_PROFILE_REPORT.md`
- `SCAIL2_SINGLE_GPU_INFERENCE_REPORT.md`
- `SCAIL2_EXPERIMENT_CONTEXT.md`
