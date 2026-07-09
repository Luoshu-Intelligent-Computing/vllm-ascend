# Gateway 服务部署指南

## 概述

本文档说明如何为 310P 128K vllm 服务接入 llm-service gateway 层，实现成熟的 thinking 控制机制。

---

## 架构

```
客户端请求 → Gateway (:8001) → vllm 容器 (:18082)
              ↓
        thinking 控制
        stop token 注入
        模型路由
```

- **vllm 容器**：直接提供推理服务，监听 `localhost:18082`
- **Gateway 层**：llm-service 提供的统一网关，监听 `localhost:8001`，负责请求预处理和响应后处理

---

## Thinking 控制机制

### 默认行为（thinking 关闭）

Gateway 自动向 vllm 请求注入 `chat_template_kwargs: {"enable_thinking": false}`，模型不进行 CoT 推理，直接输出答案。

**示例**：
```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-128k",
    "messages": [{"role":"user","content":"1+1=?"}],
    "max_tokens": 200
  }'
```

**响应**：
```json
{
  "choices": [{
    "message": {
      "content": "1 + 1 = 2",
      "reasoning": null
    },
    "finish_reason": "stop"
  }]
}
```

### 按需开启 thinking

调用方显式传 `enable_thinking: true`，启用 CoT 推理。

**示例**：
```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-128k",
    "messages": [{"role":"user","content":"解释量子纠缠"}],
    "max_tokens": 2000,
    "enable_thinking": true
  }'
```

**响应**：
```json
{
  "choices": [{
    "message": {
      "content": "量子纠缠是...",
      "reasoning": "Here's a thinking process:\n\n1. **分析用户需求**...\n2. ..."
    },
    "finish_reason": "stop"
  }]
}
```

### 兼容多种客户端写法

Gateway 自动识别以下字段，统一转换为 vllm 能识别的 `chat_template_kwargs`：

| 客户端写法 | 效果 |
|------------|------|
| `enable_thinking: true/false` | 直接识别 |
| `reasoning_effort: "high"` | 识别为开启 |
| `thinking: {"type": "enabled"}` | Anthropic 风格，识别为开启 |
| `include_reasoning: true` | 识别为开启 |

---

## 部署步骤

### 前提条件

1. vllm 容器 `vllm-qwen36-128k` 已在 `localhost:18082` 运行
2. 容器启动参数包含 `--reasoning-parser qwen3`（已确认）
3. llm-service 仓库已克隆到 `/home/nin/Workspace/worktrees/llm-service/feat-qwen35b-moe-310p-optimization-20260520`

### 一键启动（推荐）

```bash
cd /home/nin/Workspace/worktrees/llm-service/feat-qwen35b-moe-310p-optimization-20260520
./patches/310p-long-context/start-gateway.sh
```

脚本会自动：
- 检查 vllm 服务是否就绪
- 安装 llm-service 依赖（如缺失）
- 启动 gateway 在 `:8001` 端口

### 手动启动

```bash
cd /home/nin/Workspace/worktrees/llm-service/feat-qwen35b-moe-310p-optimization-20260520

# 安装依赖（仅首次）
uv sync

# 启动 gateway
PORT=8001 \
BACKENDS_CONFIG_PATH=configs/backends.310p-128k.yaml \
.venv/bin/python3 main.py
```

### 验证服务

```bash
# 检查模型列表
curl http://localhost:8001/v1/models

# 验证 thinking 默认关闭
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-128k","messages":[{"role":"user","content":"1+1=?"}],"max_tokens":50}'

# 验证 enable_thinking 开关
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-128k","messages":[{"role":"user","content":"为什么天空是蓝色的"}],"max_tokens":2000,"enable_thinking":true}'
```

---

## 配置文件

### `configs/backends.310p-128k.yaml`

Gateway 专用后端配置，指向已运行的 vllm 容器：

```yaml
backends:
  vllm-310p-128k:
    type: openai_compatible
    api_base: http://127.0.0.1:18082/v1
    timeout: 300.0
    max_retries: 1
    thinking_mode: chat_template_kwargs      # 关键：使用 chat_template_kwargs 模式
    default_enable_thinking: false           # 关键：默认关闭 thinking
    models:
      - name: qwen3.6-128k
        upstream_name: qwen3.6-128k

routing:
  strategy: model_name
  default_backend: vllm-310p-128k
```

**关键参数说明**：

- `thinking_mode: chat_template_kwargs`：告诉 gateway 用 `chat_template_kwargs` 传递 thinking 控制（适配 llama.cpp 风格 chat template）
- `default_enable_thinking: false`：默认关闭 thinking，避免过度思考/循环推理
- `api_base: http://127.0.0.1:18082/v1`：指向 vllm 容器端口

### `configs/optional.ails-a1-310p-gateway-only.yaml`

可选入口配置（仅用于文档说明，实际启动通过环境变量控制）：

```yaml
service:
  host: 0.0.0.0
  port: 8001

backends_config: "backends.310p-128k.yaml"
enabled_backends: []

routing:
  strategy: model_name
  default_backend: vllm-310p-128k
```

**注意**：`service.port` 在这里仅为文档说明，实际由环境变量 `PORT=8001` 控制。

---

## 故障排查

### 1. Gateway 启动失败：ModuleNotFoundError

**现象**：
```
ModuleNotFoundError: No module named 'llm_service'
```

**解决**：
```bash
cd /home/nin/Workspace/worktrees/llm-service/feat-qwen35b-moe-310p-optimization-20260520
uv sync
```

### 2. vllm 服务未就绪

**现象**：
```
[ERROR] vllm 服务未就绪，请先确认 vllm-qwen36-128k 容器正常运行
```

**检查**：
```bash
sudo podman ps | grep vllm-qwen36-128k
curl http://localhost:18082/v1/models
```

**修复**：
```bash
./patches/310p-long-context/start-128k-service.sh
```

### 3. thinking 仍然泄露到 content

**检查 vllm 容器启动参数**：
```bash
sudo podman inspect vllm-qwen36-128k --format '{{range .Config.Cmd}}{{println .}}{{end}}' | grep reasoning-parser
```

**必须包含**：`--reasoning-parser qwen3`

### 4. 端口冲突

**现象**：
```
OSError: [Errno 98] Address already in use
```

**检查**：
```bash
ss -tlnp | grep 8001
```

**修改端口**：
```bash
PORT=8002 BACKENDS_CONFIG_PATH=configs/backends.310p-128k.yaml .venv/bin/python3 main.py
```

---

## 技术原理

### Gateway 如何控制 thinking

1. **请求预处理**（`server.py:243-244`）：
   ```python
   apply_default_enable_thinking(payload, backend_default_enable_thinking)
   apply_thinking_mode(payload, backend_name, backend_thinking_mode)
   ```

2. **参数注入**（`thinking_mode.py:56-66`）：
   - 当 `thinking_mode == "chat_template_kwargs"` 且 `default_enable_thinking == False`
   - Gateway 向下游请求注入 `chat_template_kwargs: {"enable_thinking": false}`

3. **vllm 处理**：
   - vllm 的 Qwen chat template 读取 `chat_template_kwargs["enable_thinking"]`
   - `false` → 不生成 `<think>` 块，直接输出答案
   - `true` → 生成 `<think>` 块，`--reasoning-parser qwen3` 将其分离到 `reasoning` 字段

### 为什么不在 vllm 层默认关闭

vllm 的 `--reasoning-parser qwen3` 只负责**分离** thinking 内容到 `reasoning` 字段，不控制**是否生成**。生成行为由 chat template 的 `enable_thinking` 变量控制。

Gateway 层统一注入 `enable_thinking=false`，避免：
- 每个客户端都要手动传参
- 不同客户端写法不一致（`enable_thinking` / `reasoning_effort` / `thinking.type`）

---

## 相关文件

| 文件 | 说明 |
|------|------|
| `patches/310p-long-context/start-gateway.sh` | Gateway 一键启动脚本 |
| `patches/310p-long-context/start-128k-service.sh` | vllm 容器启动脚本 |
| `configs/backends.310p-128k.yaml` | Gateway 专用后端配置 |
| `configs/optional.ails-a1-310p-gateway-only.yaml` | 可选入口配置（文档用） |
| `src/llm_service/thinking_mode.py` | Thinking 控制核心逻辑 |
| `src/llm_service/server.py:243-244` | 请求预处理入口 |

---

## 附录：完整测试用例

### 测试 1：默认关闭 thinking（图像定位）

**请求**：
```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-128k",
    "messages": [{"role":"user","content":"图片中有一只猫，它在哪里？"}],
    "max_tokens": 100
  }'
```

**预期**：
- `content` 直接输出答案："根据提供的信息，猫在..."
- `reasoning` 为 `null`
- 无过度思考/循环推理

### 测试 2：开启 thinking（GPQA）

**请求**：
```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-128k",
    "messages": [{"role":"user","content":"证明费马大定理"}],
    "max_tokens": 4096,
    "enable_thinking": true
  }'
```

**预期**：
- `reasoning` 包含完整推理过程
- `content` 是最终结论
- `finish_reason: "stop"`

### 测试 3：兼容 Anthropic 风格

**请求**：
```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-128k",
    "messages": [{"role":"user","content":"解释相对论"}],
    "max_tokens": 2000,
    "thinking": {"type": "enabled"}
  }'
```

**预期**：
- Gateway 识别 `thinking.type` 并转换为 `chat_template_kwargs: {"enable_thinking": true}`
- `reasoning` 有内容

### 测试 4：多模态图文理解

模型原生支持图文混合输入，通过 Gateway 透传无需额外配置。

**请求（base64 编码图片）**：
```python
import base64, json, urllib.request

with open("/path/to/image.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

payload = json.dumps({
    "model": "qwen3.6-128k",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": "图中是什么场景？"}
        ]
    }],
    "max_tokens": 500
}).encode("utf-8")

req = urllib.request.Request(
    "http://localhost:8001/v1/chat/completions",
    data=payload,
    headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(req, timeout=120) as resp:
    r = json.loads(resp.read())
    print(r["choices"][0]["message"]["content"])
```

**预期**：
- `content` 直接输出图片描述（Gateway 默认 thinking=false，无推理过程泄露）
- `reasoning` 为 `null`
- 图片 token 会计入 `prompt_tokens`（通常数百 token）

**验证结果（2026-07-02）**：

测试图片为夜间城市火灾现场，模型输出（节选）：
> 这是一张从高处俯拍的夜间城市街景照片，画面中心是一场正在发生的大规模火灾，现场有多辆消防车和救援人员正在紧急处置。至少 8 辆红色消防车分布在十字路口及周边道路……

✅ 验证通过，prompt_tokens=367（含图片编码）。

---

## 版本信息

- **文档版本**：v1.0
- **更新日期**：2026-07-02
- **vllm 镜像**：registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend:26.0.0-poc-300i-duo-py311-ubuntu24.04-arm64
- **llm-service 分支**：feat/310p-opt
- **硬件**：Atlas 300I Duo（ails-a1），NPU0+NPU1
- **CANN**：9.0.T3，driver 24.1.RC3
