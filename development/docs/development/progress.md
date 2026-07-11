# 310P Long Context Fix Progress

> **⚠️ 文档状态说明**  
> 本文档记录完整的开发历程，包括已放弃的技术路线（causal_fa_310p kernel）。  
> **当前生产方案**: 动态 chunk mask（源码烘焙） + 新镜像 `310p-opt-20260708`  
> **部署指南**: 见 [310P_PRODUCTION_DEPLOYMENT.md](../guides/310P_PRODUCTION_DEPLOYMENT.md)  
> **二轮总结**: 见 [../../docs/PHASE_2_COMPLETION_SUMMARY.md](../../docs/PHASE_2_COMPLETION_SUMMARY.md)

---

## 📊 二轮最终状态（2026-07-09）

### ✅ 交付成果

**镜像构建完成**:
- Ubuntu: `registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend:310p-opt-20260708` (16 GB)
- openEuler: `registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend:310p-opt-openeuler-20260708` (16.7 GB)
- 特性: 源码改动已烘焙，GDN kernel 已编译，开箱即用

**技术方案**:
- ✅ 动态 chunk mask: 8 GB → 8 MB (-99.9% 内存)
- ✅ 128K 上下文支持（max_model_len=131072）
- ✅ Gateway 层部署（端口 8001，thinking 控制）
- ❌ causal_fa_310p kernel 已放弃（seq_len≥2048 hang bug 无法修复）

**验证数据**:
| 场景 | Prompt Tokens | 状态 |
|------|--------------|------|
| 基础推理 | 11 | ✅ content="1 + 1 = 2" |
| 长文本 | 40,018 | ✅ 正常 |
| 极长文本 | 60,015 | ✅ 正常 |
| 理论上限 | 131,072 | ✅ 支持 |

### 🔄 与一轮 POC 的差异

| 对比项 | 一轮 POC (nightly 外挂 patch) | 二轮生产 (源码烘焙) |
|--------|------------------------------|-------------------|
| 部署方式 | 容器启动时手动 cp patch + 重编译 | 镜像已包含，直接启动 |
| 代码位置 | `patches/310p-long-context/*.py` 外挂 | `vllm_ascend/_310p/attention/` 源码中 |
| 仓库结构 | llm-service 仓库中维护 patch | 310p-vllm-ascend fork 仓库 |
| 镜像标签 | `nightly-main-310p` (官方 nightly) | `310p-opt-20260708` (自建) |
| Gateway | 无 | ✅ llm-service Gateway (thinking 控制) |

### 📁 当前代码位置

**生产镜像源码**（310p-vllm-ascend fork）:
```
vllm_ascend/_310p/attention/
├── metadata_builder.py    # 修复: attn_mask = None
├── attention_v1.py         # 修复: forward_prefill_310 动态 chunk mask
└── attention_mask.py       # 修复: lazy initialization
```

**Gateway 配置**（llm-service 仓库）:
```
configs/
├── backends.310p-128k.yaml              # Backend 配置
└── optional.ails-a1-310p-gateway-only.yaml  # Gateway 入口
```

---

## 📚 历史开发记录

以下是完整的开发历程，包括已放弃的技术路线和调试过程。

---

## 目标
让 Qwen3.6-35B-A3B-w8a8 在 Ascend 310P 上支持最大 131072 token（128K）长度，消除 O(L²) attention mask OOM。

## 根本原因
`AscendAttentionMetadataBuilder.build()` 在每次推理时无条件调用
`get_attention_mask(max_model_len)` → 分配 `max_model_len × max_model_len × 2B` 的 mask。
在 65536 tokens 下 = 8 GB，超出可用显存。

## 修复方案

**动态 chunk mask 方案**：

覆盖三个关键文件，跳过预分配，改为按需动态生成：

1. `metadata_builder.py` — `build()` 设 `attn_mask = None`
2. `attention_v1.py` — `forward_prefill_310` 动态生成 `[T, T]` chunk mask（T ≤ 2048）
3. `attention_mask.py` — lazy init，`get_splitfuse_mask` 按需生成 `[T, max_seqlen]` mask

**关键优势**：
- Prefill chunk mask：T × T（T ≤ 2048）→ 最大 **8 MB**
- Chunked prefill mask：2048 × 32768 → **128 MB @ 32k**
- 临时分配，推理后释放，不占用永久显存

## 已修改文件

三个 patch 文件：
- `patches/310p-long-context/metadata_builder.py`
- `patches/310p-long-context/attention_v1.py`
- `patches/310p-long-context/attention_mask.py`

测试脚本（5个）：
- `test_integration.py` — Python 集成测试
- `test_310p_causal_fa.sh` — Shell 测试脚本
- `performance_benchmark.py` — Prefill/Decode 性能测试
- `decode_benchmark.py` — Decode 专项测试（差值法）
- `gsm8k_evaluation.py` — GSM8K 精度评估

## 容器配置（生产验证）

```bash
podman run -d --name vllm-qwen36-64k \
  --privileged --network host \
  --device /dev/davinci0 --device /dev/davinci1 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -e ASCEND_RT_VISIBLE_DEVICES=0,1 \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
  -v /usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/driver:ro \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /srv/meetai/models:/models \
  -v /path/to/patches:/workspace/patches:ro \
  registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend:26.0.0-poc-300i-duo-py311-ubuntu24.04-arm64 \
  bash -c 'source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh && \
    cp /workspace/patches/*.py $(python3 -c "import vllm_ascend; print(vllm_ascend.__path__[0])")/_310p/attention/ && \
    vllm serve /models/... --max-model-len 65536 --tp 2 ...'
```

**关键参数**:
- `--max-model-len 65536` — 支持 64k 上下文
- `--tp 2` — 双卡张量并行
- `--max-num-batched-tokens 2048` — chunked prefill batch 大小
- `--reasoning-parser qwen3` — 分离 thinking 到 reasoning 字段
- `--enable-chunked-prefill` — 启用分块 prefill
- 无需 `--enforce-eager`（patches 解决了显存问题）
--additional-config '{"ascend_compilation_config": {"fuse_norm_quant": false}}'
```

## 调试历程
1. `--device npu` 参数不存在 → 去掉
2. 缺少宿主机 driver 挂载 → aclInit 507008 错误 → 参考 vllm-32k-test 容器配置修复
3. `DynamicQuantKernelNpuOpApi` OOM → 加 `fuse_norm_quant=false`
4. 图捕获 OOM（8GB）→ 加 `--enforce-eager`
5. `build()` 签名错误（缺 `common_prefix_len`）→ 修正
6. `get_attention_mask()` 仍被调用（else 分支）→ 彻底设 `attn_mask = None`

## 当前状态（2026-06-25 12:12）

### ✅ 已解决
- OOM 修复成功：`metadata_builder.py` 覆盖 `build()`，设 `attn_mask = None`
- 短请求推理正常（seq_len=16 tokens，`causal_fa_310p` 成功返回）
- 容器启动正常，无 OOM

### ❌ 新发现阻塞：causal_fa_310p 内核 hang（seq_len=2048）

**现象：**
- 短请求（~16 tokens）：完整执行，`causal_fa_310p` 正常返回
- 长请求（chunked prefill，每 chunk 2048 tokens）：内核调用后 hang，不返回
- debug 日志停在 `query.shape: [2048, 8, 256]` 之后，`key.shape` 未打印
- 说明内核在 `causal_fa_310p(query, key, value, seq_len_cpu=[2048], scale)` 调用处挂起

**验证数据：**
- 50 次成功调用（seq_len=16，短请求的 40 层 × 多次）
- seq_len=2048 时 100% hang

**根因：** `causal_fa_310p` 内核对 seq_len=2048 的场景有 hang bug，需要内核开发方修复。

## 最终方案（2026-06-25 12:35）

### ✅ 动态小 mask 方案验证成功

放弃不稳定的 `causal_fa_310p` 内核，改用 `_npu_flash_attention` + 动态按需生成 mask：

**核心修改：`forward_prefill_310`**
```python
T = aligned_tokens  # 当前 chunk 大小，最大 2048
# 生成 [T, T] 因果 mask，最大 2048×2048 = 8 MB（相比 65536² = 8 GB）
tril = torch.ones((T, T), dtype=torch.bool, device=query.device).tril_()
chunk_mask = torch.zeros((T, T), dtype=torch.float16, device=query.device)
chunk_mask.masked_fill_(~tril, float("-inf"))
mask = torch_npu.npu_format_cast(nd_to_nz_2d(chunk_mask), ACL_FORMAT_FRACTAL_NZ)
torch_npu._npu_flash_attention(query, key, value, mask=mask, ...)
```

**测试结果：**
- ✅ 短请求（14 tokens）：正常返回
- ✅ 中等上下文（723 tokens）：正确回答（"Paris"）
- ✅ 长上下文（4015 tokens）：正常完成，无 OOM
- ✅ 长上下文（8015 tokens）：正常完成，无 OOM
- ✅ 长上下文（38533 tokens）：正常完成，正确回答
- ✅ 长上下文（50433 tokens）：正常完成，正确回答
- ✅ 长上下文（56733 tokens）：正常完成，正确回答
- ✅ **极限测试（64433 tokens）：正常完成，回答"Paris"** ← 接近 65536 上限

**三个 patch 文件最终状态：**
1. `metadata_builder.py` — `build()` 设 `attn_mask = None`（所有路径都不用 attn_metadata.attn_mask）
2. `attention_v1.py` — `forward_prefill_310` 动态生成 `[T, T]` chunk-sized 因果 mask（T ≤ 2048）
3. `attention_mask.py` — lazy init，`get_splitfuse_mask` 按需生成

**结论：** 65536 token 长上下文在 310P (2×43GB) 上可以正常推理，无 OOM，无 hang。

---

## 精度验证（2026-06-25）

### 验证状态: ✅ 已完成

**功能验证**:
- 64k 服务：14→64433 tokens 全部通过（简单问答、常识、代码、多轮、长上下文 57k）
- 32k 服务：27k token 长上下文精度正常
- 两种配置均正常，显存占用合理（~31.7 GB/卡 64k，~31.4 GB/卡 32k）

**定量精度评估 - GSM8K**:
- 数据集：GSM8K test（数学推理）
- 样本数：50
- 准确率：**98.0%（49/50）**
- 错误分析：唯一错误为边界问题（预测 12.0，正确答案 13.0），非系统性错误

**性能基线**:
- Prefill 峰值吞吐：~960 t/s（16k 规模）
- Decode 吞吐：~31.5 t/s（~31.6 ms/token）
- 32k E2E 延迟：~37.9s

**验证结论**: 
- ✅ 动态 chunk mask patches **无精度损失**
- ✅ 服务稳定，50 题无超时/异常
- ✅ 32k/64k 配置均可正常使用

**详细报告**: 
- 功能测试：`docs/reports/TEST_RESULTS_20260625_FINAL.md`
- GSM8K 评估：`docs/reports/GSM8K_EVALUATION_20260625.md`
- 性能基线：`docs/reports/PERFORMANCE_BASELINE_20260625.md`

---

## 128K 精度问题根因分析（2026-06-27）

### 问题描述
- 64K 服务（max_model_len=65536）：GSM8K 98%，精度正常
- 128K 服务（max_model_len=131072）：所有上下文长度精度异常，包括 32K、48K 短输入

### 调查过程

**Step 1：mask 构建层验证**

在容器内 NPU 上实测（Ascend310P3）：
```
64K:  nd_to_nz_spec([T, 65536])  → [1, 4096, 64, 16]，npu_format_cast OK，8.4MB
128K: nd_to_nz_spec([T, 131072]) → [1, 8192, 64, 16]，npu_format_cast OK，16.8MB
```
结论：mask 构建本身在两种尺寸下都成功，**mask 构建不是问题所在**。

**Step 2：128K 服务能正常响应**

对 128K 服务发送短请求（14 tokens）返回正常，服务本身运行正常。

**Step 3：追踪 max_model_len 的影响范围**

从代码链路看：
```
vllm serve --max-model-len 131072
  → AscendAttentionMetadataBuilder310.__init__()
    → AttentionMaskBuilder310(device, max_model_len=131072)
      → AttentionMaskBuilder310.max_seqlen = 131072  ← class variable，全局生效
```

`max_seqlen` 是类变量（class variable），一旦设置，影响所有实例、所有层、整个进程生命周期。

### 根因

**问题不在 nd_to_nz_spec 或 npu_format_cast，而在 `_npu_paged_attention_splitfuse` 对 mask 宽度的约束。**

`get_splitfuse_mask` 生成的 mask shape：
- 64K 服务：`[T, 65536]` → NZ 格式 `[1, 4096, T_pad, 16]`
- 128K 服务：`[T, 131072]` → NZ 格式 `[1, 8192, T_pad, 16]`

`_npu_paged_attention_splitfuse` 是 ATB 算子（`torch.ops.atb._npu_paged_attention_splitfuse`），其 `mask` 参数（context_len 维度）存在硬件/算子层面的最大宽度限制。

**当 max_model_len=131072 时，splitfuse mask 宽度翻倍为 131072（NZ 的 K/16=8192），超出了 ATB 算子的支持范围。** 算子不报错但计算结果错误（精度静默损坏），导致所有走 `ChunkedPrefill` / `PrefillCacheHit` 路径的请求精度异常。

**这解释了为什么短输入（32K prompt）在 128K 服务中也会失败**：问题不是输入长度，而是服务初始化时 `cls.max_seqlen=131072` 这一全局状态。只要服务是 128K 配置，splitfuse mask 就始终是 131072 宽，无论实际输入多短，一旦触发 ChunkedPrefill 路径就会走到有缺陷的 mask。

**为什么 64K 服务精度正常**：64K 时 splitfuse mask 宽度为 65536，ATB 算子在此范围内行为正确。

### 用户质疑的 RoPE 假设为何错误

RoPE 问题会随序列长度增加而恶化，64K 服务在 64K 输入时会更差。但观察到的现象是 128K 服务在 32K 短输入时也异常——这与 RoPE 无关，与 max_seqlen 全局配置有关，验证了 RoPE 假设错误。

### 修复方向

**方向 A（推荐）**：为 128K 服务的 splitfuse 路径绕过 ATB 算子限制
- 在 `get_splitfuse_mask` 中将 mask 宽度截断到实际 KV 长度（`actual_context_len`）而非 `max_seqlen`
- 或者：对 128K 场景改用 `_npu_flash_attention`（已知对 65K 下 Prefill 有效），放弃 splitfuse 路径

**方向 B**：向华为确认 `_npu_paged_attention_splitfuse` 的 mask K 维度上限

### 待验证
- ATB 算子 `_npu_paged_attention_splitfuse` 的 mask 宽度硬限制的精确值（是 65536 还是其他）
- 在 128K 服务中强制截断 mask 宽度到 65536 后精度是否恢复

### ✅ 已解决（2026-06-29）

通过实验验证：根因**不是** mask 宽度限制，而是 `max_num_batched_tokens=2048` 直接触发 ATB 算子精度问题：

- `max_num_batched_tokens=2048` → LongBench 输出乱码（0/5 正确）
- `max_num_batched_tokens=1024` → LongBench 全部正确（10/10 语义匹配）、GSM8K 98%、GPQA-Diamond 80%

**最终生产配置**：`--max-model-len 131072 --max-num-batched-tokens 1024`（见 `docs/guides/DEPLOYMENT_GUIDE.md`）

---

## 当前状态总结（2026-07-06 更新）

### ✅ 已完成

| 里程碑 | 状态 | 说明 |
|--------|------|------|
| 64K 长上下文支持 | ✅ | 动态 chunk mask 方案，14→65536 tokens 验证通过 |
| 128K 长上下文支持 | ✅ | max_model_len=131072，ARM（ails-a1）生产可用 |
| 精度问题根因定位 | ✅ | max_num_batched_tokens=2048 是根因 |
| 精度修复 | ✅ | 降至 1024，LongBench 14.02，GSM8K 98% |
| 性能基线采集 | ✅ | Prefill ~633 t/s（8K），Decode ~31.5 t/s |
| 精度评估（GPQA） | ✅ | 80%（20样本，thinking 模式） |
| **x86 部署验证（2026-07-06）** | ✅ | ails-a2 x86 服务成功启动，128K 长上下文验证通过 |
| **nightly 镜像 GDN 算子修复（2026-07-06）** | ✅ | 重编 vllm_ascend_C.so（SOC_VERSION=ascend310p1），chunk 算子注册成功 |

### 部署现状（2026-07-06）

| 服务器 | 架构 | 镜像 | 容器 | 端口 | 状态 |
|--------|------|------|------|------|------|
| ails-a1 | aarch64 | POC 26.0.0 arm64 | vllm-qwen36-128k | 18082 | ✅ 生产运行 |
| ails-a2 | x86_64 | POC 26.0.0 x86 | vllm-qwen36-128k | 18082 | ✅ 2026-07-06 验证 |

### ⏳ 待推进

| 项目 | 优先级 | 说明 |
|------|--------|------|
| causal_fa_310p hang bug | P1 | seq_len≥2048 时 100% hang，参考 vllm-ascend PR #9458 CATLASS_UNIFIED_CORE 路径 |
| nightly 服务性能对比（T1）| P1 | ails-a2 x86 服务已启动，需与 ails-a1 633 t/s 基线对比 |
| Prefill 性能恢复 | P2 | max_batched=1024 导致吞吐下降 34%，causal_fa 修复后可改善 |
| causal_fa 向量化优化 | P3 | 标量 GEMM（效率约 1/1333），修复 hang 后做向量化 |

---

## nightly patch 接口适配（2026-07-07）

### 实施记录
- [完成] `attention_mask.py`：`get_attention_mask` 签名修复 `(causal: bool, model_config)` — `patches/310p-long-context-nightly/attention_mask.py`
- [完成] `attention_mask.py`：`get_splitfuse_mask` 替换为 on-the-fly broadcasting（修复 128K OOM） — `patches/310p-long-context-nightly/attention_mask.py`
- [完成] `metadata_builder.py`：移除错误的 `attn_metadata.attn_mask = None`（破坏 forward_prefill_310 的 mask 传入） — `patches/310p-long-context-nightly/metadata_builder.py`
- [完成] 语法验证：两个文件在 nightly 容器内 python3 ast.parse 通过

### 关键发现

**nightly 容器（CANN 9.1.0-beta.1）`is_compressed_mask_supported()` 返回 False**

`_npu_flash_attention_v3` 和 `_npu_paged_attention_splitfuse_v2` 均不存在于此版本，因此：
- forward_prefill_310 走旧路径：`_npu_flash_attention` + FRACTAL_NZ mask
- forward_chunked_prefill_310 走旧路径：`_npu_paged_attention_splitfuse` + get_splitfuse_mask

### 变更说明

| 文件 | 变更 | 原因 |
|------|------|------|
| `attention_mask.py` | `get_attention_mask(self, causal: bool, model_config)` | nightly 父类调用改为 2 参数；310P mask 不区分 causal 标志，忽略该参数 |
| `attention_mask.py` | `get_splitfuse_mask` 改用 broadcasting on-the-fly | nightly 实现预分配 `[max_seqlen, max_seqlen]`，128K 下 = 34GB；broadcasting 方案生成 `[T, 131072]`（T≤1024），约 256MB 临时分配 |
| `metadata_builder.py` | 移除 `attn_metadata.attn_mask = None` | nightly 的 `forward_prefill_310` 直接读 `attn_metadata.attn_mask` 传给 `_npu_flash_attention`；置 None 会导致 NoneType 传入算子崩溃；OOM 修复已在 `attention_mask.py` 通过 cap 到 2048 完成 |

### 自验证结果
- 编译（语法）：通过（nightly 容器内 ast.parse）
- 推理：待容器挂载测试

### 当前代码状态
- `patches/310p-long-context-nightly/attention_mask.py`：nightly 接口兼容，OOM 已修复（两条路径）
- `patches/310p-long-context-nightly/metadata_builder.py`：接口兼容，不破坏 mask 传入链路
- `patches/310p-long-context-nightly/` 目录无 `attention_v1.py`（nightly 版接口未变，不需要 patch）
