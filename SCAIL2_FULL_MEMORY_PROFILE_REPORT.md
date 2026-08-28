# SCAIL2 完整推理分阶段显存报告

日期：2026-08-28

## 1. 结论

本次在物理 GPU 2、3 上完整执行 4 个 segment、每段 6 个 diffusion step，并同时采集 PyTorch allocator、CUDA device used 和 200 ms NVML 监控数据。

此前报告中的 `23193.4 MiB` 是 **segment 1 / step 1 / block 0** 的 `torch.cuda.max_memory_allocated()`，不是整个推理任务的设备显存峰值。完整运行得到：

| 范围 | 峰值 | 位置 |
|---|---:|---|
| block 0 allocated | 23277.4 MiB | segment 2/3，step 6，block 0 |
| 完整 DiT allocated | **23754.3 MiB** | segment 2/3，step 6，block 1--39 |
| 完整 DiT reserved | **26342.0 MiB** | 81 帧 segment 的 allocator 平台 |
| 完整 DiT device/NVML used | **27467 MiB** | 两张卡的 81 帧 DiT 计算阶段 |
| 完整任务 GPU 2 NVML used | **29725 MiB** | rank 0，segment 1 VAE decode |
| 完整任务 GPU 3 NVML used | **27467 MiB** | rank 1，DiT 计算阶段 |

因此，监控程序看到约 27G 是正确的：它对应完整 DiT 的设备级占用，而不是 block 0 的 PyTorch live tensor。整个任务的最高点实际出现在仅由 rank 0 执行的 VAE decode，约 29.7 GiB。

## 2. 实验条件

- GPU：物理 GPU 2、3，A100-SXM4-40GB。
- 双卡 FSDP `FULL_SHARD`，每个 rank 对应一张物理 GPU。
- 模型：SCAIL-14B BF16，T5/CLIP 使用预计算 conditioning cache。
- 输入：`testdata/101`，512×896，297 帧。
- segment：81、81、81、57 帧；每个 segment 6 个 diffusion step。
- FFN chunk size：8192。
- RoPE chunk size：8192，内部保持 FP64/complex128。
- allocator：`expandable_segments:True`。
- 新增 `--full-memory-profile`：不截断推理，完整执行 VAE decode 和输出。
- NVML：父进程每 200 ms 采集一次物理 GPU `memory.used`。

实验日志：`experiment_logs/fsdp_baseline/101-20260828-155713.log`

完整输出：`experiment_outputs/fsdp_baseline/101-20260828-155713.mp4`

## 3. 三种显存口径

| 字段 | 含义 |
|---|---|
| allocated | 当前存活或阶段内曾存活的 PyTorch tensor，由 PyTorch allocator 统计 |
| reserved | PyTorch allocator 已从 CUDA driver 取得的显存，包括 allocated 和暂未复用的缓存块 |
| device/NVML used | 整张卡实际占用，包括 reserved、CUDA context、NCCL、cuBLAS、kernel workspace 和驱动分配 |

近似关系为：

```text
device used = live allocated + allocator cached/unallocated + CUDA/NCCL/driver
            = reserved       + CUDA/NCCL/driver
```

本次两张卡在主要计算阶段的 `device used - reserved` 稳定为约 `1124.8 MiB`。这部分不出现在 `torch.cuda.memory_allocated()` 中。

## 4. 完整流程分阶段总表

所有数值单位均为 MiB。`allocated 峰值` 是阶段内 PyTorch live tensor 的最高值；`同步 device used` 是阶段边界同步后由 `cudaMemGetInfo` 得到的整卡占用，并由 NVML 采样交叉验证。

| 阶段 | Rank/GPU | allocated 峰值 | reserved | 同步 device used | 主要内容 |
|---|---|---:|---:|---:|---|
| Engine ready | 两个 rank | 16120.1 | 18216 | 19318.8 | DiT FSDP shard、VAE、运行时基础占用 |
| Reference VAE encode | 两个 rank | 18361.4 | 18718 | 19826.8 | 参考图 VAE encode 临时激活 |
| Conditioning ready | 两个 rank | 16167.6 当前值 | 18718 | 19826.8 | T5/CLIP cache tensor 已上传 GPU |
| Segment 1 prepare | 两个 rank | 18417.1 | 18980 | 20088.8 | pose VAE encode、mask、noise、条件 tensor |
| Segment 2/3 prepare | 两个 rank | 18895.3 | 19560 | 20684.8 | 额外包含 history latent/mask |
| Segment 4 prepare | rank 0 | 18736.5 | 19420 | 20544.8 | 57 帧短 segment 条件准备 |
| 81 帧 DiT，segment 1 | 两个 rank | 23734.1 | 26342 | 27466.8 | 40 个 DiT block；后续 block 使用 FP32 residual |
| 81 帧 DiT，segment 2/3 | rank 0 | **23754.3** | 26282 | 27406.8 | 完整 DiT live tensor 最高点 |
| 81 帧 DiT，segment 2/3 | rank 1 | 23739.1 | 26282 | 27406.8 | rank 间 live tensor 最大差约 15.2 |
| 57 帧 DiT，segment 4 | rank 0 | 22140.8 | 24004 | 25128.8 | token 数减少，峰值下降约 1.61 GiB |
| Scheduler step | 两个 rank | 最高约 17431.6 | 最高 26342 | 最高 27466.8 | scheduler 本身占用低，但 allocator 保留 DiT 缓存池 |
| Segment cleanup | 两个 rank | 当前约 16654.5 | 约 16798 | 17922.8 | `empty_cache()` 后大部分 DiT 缓存释放 |
| VAE decode，segment 1 | rank 0/GPU 2 | **25617.3** | **28600** | **29724.8** | 全任务最高峰；rank 1 不执行 decode |
| VAE decode，segment 2/3 | rank 0/GPU 2 | 25617.3 | 28040 | 29164.8 | 81 帧 decode |
| VAE decode，segment 4 | rank 0/GPU 2 | 25483.4 | 27900 | 29024.8 | 57 帧 decode |
| History VAE encode | rank 0/GPU 2 | 18841.7 | 19000 | 20124.8 | 1 帧 overlap 重新编码 |
| Inference complete | rank 0 | 16597.8 当前值 | 16714 | 17838.8 | 输出已转 CPU，GPU 临时量释放 |
| Inference complete | rank 1 | 16597.8 当前值 | 16734 | 17858.8 | 等待 rank 0 输出流程 |

## 5. 为什么 DiT 监控值约为 27G

以 segment 1 后期 step 的稳定平台为例：

| 组成 | 大小 |
|---|---:|
| PyTorch live allocated | 23734.1 |
| allocator reserved 但当时未被 live tensor 使用 | 2607.9 |
| CUDA context、NCCL、库和驱动分配 | 1124.8 |
| **设备实际 used** | **27466.8** |

计算关系：

```text
23734.1 + (26342.0 - 23734.1) + (27466.8 - 26342.0)
= 27466.8 MiB
```

也就是说，报告中的 live tensor 峰值与监控值之间约 3.73 GiB 的差距，由约 2.61 GiB allocator 缓存和约 1.10 GiB 非 PyTorch CUDA/NCCL/驱动占用组成。

## 6. 为什么 VAE decode 达到约 29.7G

segment 1 的 VAE decode 是整个任务的最高点：

| 组成 | 大小 |
|---|---:|
| PyTorch live allocated 峰值 | 25617.3 |
| allocator cached/unallocated | 2982.7 |
| CUDA context、NCCL、库和驱动分配 | 1124.8 |
| **设备实际 used / NVML 峰值** | **29724.8 / 29725** |

VAE 只在 rank 0 解码，rank 1 同期不产生 VAE activation。因此物理 GPU 2 的全任务峰值为 29725 MiB，而物理 GPU 3 的全任务峰值仍是 DiT 阶段的 27467 MiB。

VAE decode 开始前已经执行 `gc.collect()`、`torch.cuda.synchronize()` 和 `torch.cuda.empty_cache()`；开始时 rank 0 allocated 为约 16654.5 MiB。decode 期间额外产生约：

```text
25617.3 - 16654.5 = 8962.8 MiB
```

因此 VAE 峰值不是前一个 DiT allocator 缓存未清理造成的，而是 replicated VAE decode 激活与仍常驻的 DiT FSDP shard 叠加造成的真实峰值。

### 6.1 单 segment VAE decode 精细剖析

后续单 segment 实验曾用临时埋点把 VAE decode 细化到每个 latent 时间步、分辨率、causal cache、temporal cat、padding、卷积 workspace、空间上采样和输出累计。全部明细单独整理在 `SCAIL2_VAE_MEMORY_PROFILE_REPORT.md`；对应日志为 `experiment_logs/fsdp_baseline/101-20260828-173206.log`。数据采集完成后，临时 profiling 入口和 VAE 内部埋点已从运行时代码移除。

## 7. DiT block 0 与后续 block 的差异

segment 1 / step 1 的实测值：

| 指标 | Block 0 | Block 1--39 | 差值 |
|---|---:|---:|---:|
| Block 入口 allocated | 19196.8 | 19673.7 | +476.9 |
| Block 结束 allocated | 20627.6 | 21104.4 | +476.8 |
| Block 内最高 allocated | 23193.4 | 23670.3 | **+476.9** |

原因是 block 0 的初始 hidden 为 BF16；完成第一个 residual 后，hidden 变为 FP32。后续 39 个 block 因此始终比 block 0 多保留一份约 476.9 MiB 的长序列 hidden。

原 DiT 显存报告对这一点已有文字说明，但主表只展示了 block 0。本次完整剖析确认这 476.9 MiB 会直接抬高完整 DiT 的 live allocated 峰值。

## 8. 每个 segment / step 的 DiT 峰值

下表取两个 rank 中更高的 block 峰值；单位为 MiB。

| Segment | Step 1 | Step 2 | Step 3 | Step 4 | Step 5 | Step 6 | Segment 最大 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1（81 帧） | 23670.3 | 23703.8 | 23719.0 | 23719.0 | 23734.1 | 23734.1 | 23734.1 |
| 2（81 帧） | 23675.3 | 23708.9 | 23708.9 | 23724.0 | 23739.1 | **23754.3** | **23754.3** |
| 3（81 帧） | 23675.3 | 23708.9 | 23708.9 | 23724.0 | 23739.1 | **23754.3** | **23754.3** |
| 4（57 帧） | 22067.1 | 22095.4 | 22095.4 | 22110.5 | 22125.7 | 22140.8 | 22140.8 |

同一 segment 内峰值缓慢增加约 64--79 MiB，来自 sampling loop 中逐步变化或保留的 scheduler/latent 状态，而不是某个 DiT block 单独变大。对于同一个 step，block 1--39 的峰值基本相同。

## 9. 结果一致性与剖析开销

完整任务成功生成 297 帧、512×896、30 fps、9.9 秒视频。输出大小为 19587416 bytes。

优化前后完整 MP4 SHA-256 均为：

```text
7039c5f231eb64b544c4aa288ea5107411c9e7f51bdcf4c93d125d6e1610680a
```

说明同步采样和 block 统计没有改变最终输出。

本次从 ready/start 到 finished 为 425.537 秒；普通完整回归为 424.714 秒，剖析增加约 0.823 秒。该差值只是单次运行对照，不能作为稳定性能结论。

## 10. 后续优化含义

需要区分两个目标：

1. **降低 DiT live tensor**：当前完整 DiT allocated 峰值为 23754.3 MiB。应继续处理后续 block 的 FP32 residual、Self/Cross Q/K/V 和 RoPE/FlashAttention 附近的平台峰值。
2. **降低整条任务的物理 GPU 峰值**：当前最高点是 rank 0 VAE decode 的 29725 MiB。即使继续降低 DiT block 峰值，若不处理 VAE decode，整条任务的 NVML 峰值也不会下降。

如果目标是单卡主流程，VAE decode 与 DiT 参数常驻重叠是新的重要问题。候选方向包括在 decode 前卸载/释放 DiT resident shard、VAE tiled decode 或把 decode 放到独立阶段/设备。实施前需要分别评估额外传输时间、FSDP 生命周期和结果一致性。

## 11. 复现实验

```bash
python run_fsdp_experiment.py \
  --conditioning-cache experiment_cache/conditioning/101.safetensors \
  --physical-gpus 2,3 \
  --ffn-chunk-size 8192 \
  --rope-chunk-size 8192 \
  --expandable-segments \
  --full-memory-profile
```

`--full-memory-profile` 与会截断到首个 diffusion step 的 `--memory-probe` 互斥。
