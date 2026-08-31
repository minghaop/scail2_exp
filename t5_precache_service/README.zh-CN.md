# T5 Precache 服务使用说明

## 1. 服务用途

T5 Precache 服务常驻加载 T5 Encoder，根据输入 prompt 生成可直接交给
`scail2_inference` 使用的 `.safetensors` 缓存文件。

服务内部以规范化 prompt 的 SHA-256 作为文件索引：

- 首次提交 prompt：执行 T5 编码并写入缓存；
- 再次提交相同 prompt：不执行 T5，直接返回已有文件；
- 请求串行处理，不支持并发推理；
- negative prompt 固定为空字符串。

## 2. 运行要求

- Conda 环境：`scail2-single-gpu`
- 单张可用 NVIDIA GPU
- T5 模型目录，例如：`/raid/scail-2-20260819/umt5-xxl`
- 持久化、可写的缓存目录

T5 模型目录至少应包含：

```text
models_t5_umt5-xxl-enc-bf16.pth
tokenizer.json
tokenizer_config.json
special_tokens_map.json
spiece.model
```

## 3. 工程目录

```text
t5_precache_service/
├── run_service.py
├── t5_precache_service/
│   ├── __init__.py
│   ├── __main__.py
│   ├── service.py
│   ├── engine.py
│   └── database.py
├── README.zh-CN.md
└── work/
    ├── cache/
    └── logs/
```

`work/` 已加入该子工程自己的 `.gitignore`。未显式传入 `--cache-dir` 时，
服务默认使用 `t5_precache_service/work/cache`。

缓存格式的共享读写和校验代码位于 `scail2_inference/conditioning.py`，主推理与
本服务共同使用该实现。

## 4. 启动服务

进入 T5 Precache 子工程目录执行：

```bash
cd /raid/scail2_exp/t5_precache_service

mkdir -p work/cache work/logs

/home/panminghao/miniconda3/envs/scail2-single-gpu/bin/python \
  -u run_service.py \
  --gpu 2 \
  --host 127.0.0.1 \
  --port 8001 \
  --checkpoint-dir /raid/scail-2-20260819/umt5-xxl \
  --cache-dir work/cache \
2>&1 | tee work/logs/manual.log
```

主要参数：

| 参数 | 含义 |
|---|---|
| `--gpu` | 物理 GPU 编号，必填；服务只向 CUDA 暴露这一张卡 |
| `--host` | 监听地址；仅限本机使用时建议设为 `127.0.0.1` |
| `--port` | HTTP 端口，默认 `8001` |
| `--checkpoint-dir` | 直接指向 `umt5-xxl` 目录 |
| `--cache-dir` | 持久化缓存目录 |
| `--profile` | 推理配置，默认 `scail2-512p-bf16-v1` |

看到以下日志表示服务可以接收请求：

```text
SCAIL2_T5_SERVICE status=ready physical_gpu=2
```

前台运行时按 `Ctrl+C` 停止服务。

## 5. 生成或读取缓存

服务只提供一个缓存请求接口：

```text
POST /v1/t5-cache
```

请求示例：

```bash
curl --fail \
  -D /tmp/t5-response.headers \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"A person walking naturally"}' \
  http://127.0.0.1:8001/v1/t5-cache \
  --output conditioning.safetensors
```

响应正文就是 `.safetensors` 文件，不是 JSON。该文件包含：

```text
text_context
negative_context
```

两个 Tensor 都位于 CPU、采用 BF16，并符合 `scail2-t5-cache-v1` schema。

响应头包含：

```text
X-SCAIL2-Prompt-SHA256: <prompt hash>
X-SCAIL2-Cache-Hit: false
```

其中：

- `false`：本次执行了 T5 编码并创建文件；
- `true`：缓存已经存在，本次直接返回已有文件。

服务不提供按 hash 查询接口。需要再次获取已有结果时，重新提交相同 prompt。

## 6. 健康检查

```bash
curl --fail http://127.0.0.1:8001/health
```

返回内容包括服务状态、模型配置、CUDA 设备、缓存目录、缓存文件数量和总大小。

## 7. 缓存规则

输入 prompt 会先去掉首尾空白，再计算 SHA-256。中间的空格和换行不会在哈希前折叠。
规范化后的 prompt 原文也会写入 Safetensors metadata，因此缓存文件不应被视为匿名数据。

缓存文件路径为：

```text
<cache-dir>/<hash 前两位>/<完整 hash>.safetensors
```

例如：

```text
t5_precache_service/work/cache/95/95cf...e1ab0.safetensors
```

缓存文件还记录以下身份信息：

- 规范化后的原始 prompt；
- profile 名称；
- T5 最大文本长度；
- T5 checkpoint 文件名、大小和修改时间；
- prompt hash；
- 空 negative prompt 的 hash。

缓存命中时会验证文件结构、Tensor dtype/shape 和上述元数据。文件损坏或与当前模型配置不一致时，会重新生成。

## 8. 交给主推理流程

下载得到的文件可以直接作为 `scail2_inference` 推理任务的
`t5_cache_path`，无需再加载 T5：

```python
InferenceJob(
    # 其他输入省略
    t5_cache_path=Path("conditioning.safetensors"),
)
```

缓存文件必须由与推理服务相同的 profile 和 T5 checkpoint 生成，否则主推理流程会拒绝载入。

## 9. 常见问题

### 服务启动时报 CUDA 设备数量错误

确认 `--gpu` 指定的物理 GPU 存在且当前可用。服务会通过
`CUDA_VISIBLE_DEVICES` 只暴露指定设备，并要求进程内恰好看到一张 GPU。

### 返回 HTTP 422

prompt 为空、只包含空白，或者请求 JSON 不符合格式。正确格式为：

```json
{"prompt": "text"}
```

### 返回 HTTP 500

检查服务日志。常见原因包括模型文件缺失、缓存目录不可写、缓存文件损坏或 GPU 显存不足。

### 是否可以同时提交多个请求

当前服务设计为单 worker 串行执行。调用方应等待当前请求完成后再提交下一次请求。
