# 310P 128K 生产部署方案

**日期**: 2026-07-09  
**状态**: ✅ 生产就绪  
**镜像**: `registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend:310p-opt-20260708`

---

## 一、方案概述

### 核心改进

1. **动态 chunk mask OOM 修复**（vllm-ascend 源码烘焙）
   - 从 8 GB 预分配优化为 8 MB 动态生成
   - 支持 128K 上下文（max_model_len=131072）
   - 代码位置：`vllm_ascend/_310p/attention/metadata_builder.py`

2. **镜像构建优化**
   - 源码改动直接在 vllm-ascend fork 中
   - git submodules 初始化（catlass）
   - 无需容器启动时手动 patch 和重编译

3. **Gateway 统一入口**
   - 默认关闭 thinking，按需开启
   - 屏蔽底层 reasoning parser 复杂性
   - 端口：8001

---

## 二、架构图

```
客户端 → Gateway (8001) → vLLM (18082) → 310P NPU (TP=2)
          ↓
     thinking 控制
     (enable_thinking: true/false)
```

---

## 三、部署步骤

### 3.1 拉取镜像（Ubuntu 版）

```bash
sudo podman pull registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend:310p-opt-20260708
```

**镜像信息**：
- 大小：14.91 GB
- 基础：CANN 9.1.0-beta.1 + Ubuntu 22.04
- 包含：vLLM 0.23.0 + vllm-ascend + 310P 优化 + GDN kernel

### 3.2 启动 vLLM 服务

```bash
sudo podman run -d \
  --name vllm-310p-opt-128k \
  --privileged --network host \
  --device /dev/davinci0 --device /dev/davinci1 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -e ASCEND_RT_VISIBLE_DEVICES=0,1 \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
  -v /srv/meetai/models:/models:ro \
  registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend:310p-opt-20260708 \
  bash -c '
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export ASCEND_CUSTOM_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom
    
    python3 -m vllm.entrypoints.openai.api_server \
      --model /models/llm-service/vllm-ascend/Qwen3.6-35B-A3B-w8a8 \
      --served-model-name qwen3.6-128k \
      --host 0.0.0.0 --port 18082 \
      -tp 2 \
      --max-model-len 131072 \
      --max-num-seqs 1 \
      --max-num-batched-tokens 1024 \
      --gpu-memory-utilization 0.75 \
      --dtype float16 \
      --kv-cache-dtype auto \
      --trust-remote-code \
      --enable-chunked-prefill \
      --no-enable-prefix-caching \
      --reasoning-parser qwen3 \
      --additional-config "{\"ascend_compilation_config\": {\"fuse_norm_quant\": false}}" \
      --compilation-config "{\"cudagraph_mode\": \"FULL_DECODE_ONLY\", \"cudagraph_capture_sizes\": [1]}" \
      --async-scheduling \
      --mamba-ssm-cache-dtype float16 \
      --allowed-local-media-path /
  '
```

**关键参数说明**：
- `--max-model-len 131072`：128K 上下文窗口
- `--max-num-batched-tokens 1024`：chunked prefill 大小
- `--reasoning-parser qwen3`：解析思考过程（需配合 Gateway 使用）
- `-tp 2`：2 卡 Tensor Parallel

### 3.3 启动 Gateway

```bash
cd /home/nin/Workspace/worktrees/llm-service/feat-qwen35b-moe-310p-optimization-20260520

# 确保配置正确
cat configs/backends.310p-128k.yaml

# 启动 Gateway
PORT=8001 BACKENDS_CONFIG_PATH=configs/backends.310p-128k.yaml \
nohup .venv/bin/python main.py > /tmp/gateway_8001.log 2>&1 &

# 健康检查
curl http://localhost:8001/health
```

---

## 四、功能验证

### 4.1 基础推理

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-128k",
    "messages": [{"role": "user", "content": "1+1=?"}],
    "max_tokens": 100
  }' | jq .choices[0].message.content
```

**预期输出**：`"1 + 1 = 2"`

### 4.2 Thinking 控制

```bash
# 开启 thinking
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-128k",
    "messages": [{"role": "user", "content": "1+1=?"}],
    "max_tokens": 300,
    "enable_thinking": true
  }' | jq '{content: .choices[0].message.content, reasoning_len: (.choices[0].message.reasoning | length)}'
```

**预期输出**：
```json
{
  "content": "\n\n1 + 1 = 2",
  "reasoning_len": 692
}
```

### 4.3 长文本推理（128K 验证）

```python
import requests

# 生成 60K tokens 文本
long_text = "测试内容。" * 10000

resp = requests.post("http://localhost:8001/v1/chat/completions", json={
    "model": "qwen3.6-128k",
    "messages": [{"role": "user", "content": f"{long_text}\n\n总结："}],
    "max_tokens": 100
}, timeout=180)

result = resp.json()
print(f"Prompt tokens: {result['usage']['prompt_tokens']}")
print(f"Content: {result['choices'][0]['message']['content'][:100]}")
```

**预期 prompt_tokens**：40,000+

---

## 五、性能指标

| 指标 | 数值 |
|------|------|
| 模型 | Qwen3.6-35B-A3B-w8a8 |
| 最大上下文 | 131,072 tokens |
| Tensor Parallel | 2 |
| KV Cache | 473,875 tokens |
| 模型权重 | 18.86 GB per TP rank |
| 启动时间 | ~2 分钟 |
| Prefill 吞吐 | 1.6 tokens/s (首次) |
| Decode 吞吐 | 5.0 tokens/s |

---

## 六、故障排查

### 6.1 服务启动失败

```bash
# 查看容器日志
sudo podman logs vllm-310p-opt-128k | tail -50

# 检查 NPU 状态
npu-smi info
```

### 6.2 Gateway 返回 "upstream endpoint not found"

**原因**：vLLM 实际模型名与 Gateway 配置不匹配

**解决**：
1. 查看 vLLM 实际模型名：`curl http://localhost:18082/v1/models`
2. 更新 `configs/backends.310p-128k.yaml` 中的 `upstream_name`
3. 重启 Gateway

### 6.3 Reasoning parser 导致 content 为 null

**原因**：`max_tokens` 太小，模型在思考过程中被截断

**解决**：
- 方案 1：增加 `max_tokens` ≥ 200
- 方案 2：使用 Gateway 默认关闭 thinking
- 方案 3：直接调用 `/v1/completions` 接口（绕过 reasoning parser）

---

## 七、镜像构建记录

### 7.1 Ubuntu 版本

**构建命令**：
```bash
cd /home/nin/Workspace/310p-vllm-ascend

sudo podman build \
  -f Dockerfile.310p \
  --network host \
  --build-arg SOC_VERSION=ascend310p1 \
  --build-arg PIP_INDEX_URL="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple" \
  --build-arg http_proxy=http://127.0.0.1:10000 \
  --build-arg https_proxy=http://127.0.0.1:10000 \
  -t registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend:310p-opt-20260708 \
  .
```

**构建时间**：~40 分钟  
**状态**：✅ 成功

### 7.2 openEuler 版本

**基础镜像拉取命令**（见下方）

**构建命令**：
```bash
sudo podman build \
  -f Dockerfile.310p.openEuler \
  --network host \
  --build-arg SOC_VERSION=ascend310p1 \
  --build-arg PIP_INDEX_URL="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple" \
  --build-arg http_proxy=http://127.0.0.1:10000 \
  --build-arg https_proxy=http://127.0.0.1:10000 \
  -t registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend:310p-opt-openeuler-20260708 \
  .
```

**状态**：⏳ 待完成

---

## 八、相关文档

- 源码仓库：`/home/nin/Workspace/310p-vllm-ascend`
- Gateway 配置：`configs/backends.310p-128k.yaml`
- 启动脚本参考：`/home/nin/Workspace/310p-vllm-ascend/nightly/scripts/start-gateway.sh`
- 历史开发记录：`patches/310p-long-context/docs/development/progress.md`

---

## 九、生产部署建议

### 推荐方案（Gateway + vLLM）

- **客户端入口**：`http://localhost:8001/v1` (Gateway)
- **优势**：
  - 默认关闭 thinking，用户体验更好
  - 按需开启 CoT（`enable_thinking: true`）
  - 统一配置管理

### 备选方案（直连 vLLM）

- **直连入口**：`http://localhost:18082/v1`
- **适用场景**：
  - 性能基准测试
  - 调试 reasoning parser 行为
  - 简化部署（无需 Gateway）

### Thinking 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_thinking` | `false` | Gateway 默认关闭 |
| `max_tokens` | - | 建议 ≥ 200（开启 thinking 时） |
| `reasoning_parser` | `qwen3` | vLLM 启动参数 |

---

**文档维护者**: Claude Code  
**最后更新**: 2026-07-09
