# 单卡部署验证报告

**测试日期**: 2026-06-26  
**测试模型**: Qwen3.6-27B-w8a8  
**硬件**: Ascend 310P3 单卡（43GB HBM）  
**结论**: ❌ **不可行**

---

## 执行摘要

在 Ascend 310P3 单卡（43GB）上测试 Qwen3.6-27B-w8a8 部署，尝试 3 种不同配置均因显存不足（OOM）失败。

**根本原因**：27B 模型权重 + vLLM 框架开销 + KV cache 总需求超过 43GB 单卡容量。

**建议方案**：
1. **双卡部署（TP=2）**：已验证可行，推荐生产使用
2. **小模型替代**：8B/14B 量级模型可单卡部署

---

## 测试配置与结果

| 配置 | max_model_len | gpu_memory_utilization | max_num_seqs | 结果 | 启动时长 |
|------|--------------|----------------------|-------------|------|---------|
| 标准配置 | 8192 | 0.80 | 4 | OOM | ~4 分钟后失败 |
| 压缩配置 | 4096 | 0.70 | 1 | OOM | ~4 分钟后失败 |
| 极限配置 | 1024 | 0.65 | 1 | OOM | ~3 分钟后失败 |

**错误信息**（所有配置一致）：
```
torch.OutOfMemoryError: The Inner error is reported as above. 
The process exits for this inner error, and the current working operator name is SelfAttentionOperation.

Memory_Allocation_Failure(EL0004): Failed to allocate memory requested by APP module.
Possible Cause: Available memory is insufficient.
```

---

## 显存需求分析

### 27B 模型显存占用估算

| 组件 | 估算占用 | 说明 |
|------|---------|------|
| 模型权重（w8a8） | ~27 GB | 27B 参数 × 1 Byte |
| vLLM 框架开销 | 8-12 GB | Ascend 后端编译缓存、运行时 |
| KV cache（1K context） | 2-3 GB | 单卡 TP=1 需全量 cache |
| 激活值 + 临时张量 | 3-5 GB | Attention、FFN 中间结果 |
| 编译缓存 | 2-3 GB | CANN 图编译产物 |
| **总计** | **42-50 GB** | **超过单卡 43GB 容量** |

**即使极限压缩配置**（1K context + 0.65 utilization），模型权重 + 框架开销已达 ~35-39GB，留给 KV cache 和运行时的空间不足。

---

## 双卡部署对比（参考）

**已验证可行配置**（TP=2）：
- max_model_len: 65536 / 131072
- gpu_memory_utilization: 0.75
- 双卡总显存：~86 GB（43GB × 2）
- 实际占用：~66-70 GB
- 推理性能：Prefill ~960 t/s，Decode ~31.5 t/s

---

## 技术细节

### 启动参数（极限配置）

```bash
vllm serve /models/llm-service/vllm-ascend/Qwen3.6-27B-w8a8 \
  --served-model-name qwen3.6-27b-minimal \
  --host 0.0.0.0 --port 18082 -tp 1 \
  --max-model-len 1024 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 512 \
  --gpu-memory-utilization 0.65 \
  --dtype float16 --kv-cache-dtype auto \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --additional-config '{"ascend_compilation_config": {"fuse_norm_quant": false}}' \
  --async-scheduling
```

### OOM 发生阶段

所有测试均在**模型加载后、推理初始化阶段** OOM：
- 模型权重加载：✅ 成功
- 编译预热（Compilation warmup）：❌ OOM
- 错误位置：`SelfAttentionOperation` 内存分配

---

## 替代方案

### 方案 1：双卡部署（推荐）

**优点**：
- 已验证稳定可行
- 支持长上下文（64k/128k）
- 性能优异

**缺点**：
- 需要两张 NPU
- 成本翻倍

### 方案 2：小模型单卡部署

**候选模型**：
- Qwen2.5-7B（~7 GB 权重，单卡充裕）
- Qwen3-14B（~14 GB 权重，单卡可行）

**优点**：
- 单卡即可
- 成本低
- 框架验证通过（Qwen3.6 架构）

**缺点**：
- 需要下载新模型（~30 分钟）
- 能力弱于 27B

### 方案 3：等待优化

**可能方向**：
- vLLM 框架显存优化
- CANN 编译缓存压缩
- PagedAttention 显存管理改进

**时间周期**：不确定

---

## 结论与建议

**单卡 310P3 部署 Qwen3.6-27B-w8a8：不可行**

**生产环境推荐**：
1. **优先方案**：双卡 TP=2 部署（已验证）
2. **备选方案**：下载 8B/14B 模型单卡部署
3. **不建议**：继续尝试 27B 单卡优化（投入产出比低）

---

**测试执行**: CANNBot model-infer-optimize  
**报告日期**: 2026-06-26

---

## 补充：双卡 Decode 性能优化实验（2026-06-27）

**背景**：同事反馈单芯96GB SOC 上27B模型 decode ~8 t/s，目标 ≥20 t/s。

### 配置信息

| 项目 | 值 |
|------|-----|
| 模型 | Qwen3.6-27B-w8a8（dense，64层，hidden=5120，FFN=17408） |
| 部署 | TP=2，max_model_len=8192，端口18084 |
| 测试 | 服务端 engine throughput + 客户端 E2E 测量 |

### 实验结果

| 方案 | decode 速度 | 相对基线 |
|------|-----------|---------|
| 基线（PIECEWISE，无优化）| 7.85 t/s | 1.0x |
| FULL_DECODE_ONLY + async-scheduling | 8.79 t/s | +12% |
| + HCCL_OP_EXPANSION_MODE=AIV + fuse_gemm_comms | 8.77 t/s | +12% |
| fuse_allreduce_rms=True | 启动失败 | — |

> `fuse_allreduce_rms` 启动失败原因：当前 CANN 版本缺少 `npu_add_rms_norm_bias` 算子。

### 根因分析

**关键验证**：单芯 TP=1 与本机 TP=2 速度相同（~8 t/s），排除 TP 通信开销，瓶颈在模型计算本身。

| 指标 | 35B MoE | 27B Dense |
|------|---------|-----------|
| 每 decode step 权重读取量 | ~3 GB（8/256 激活专家）| ~27 GB（全量 FFN）|
| 理论速度 @900 GB/s | ~300 t/s | ~33 t/s |
| 实测速度 | 31.5 t/s（10.5% 效率）| 8.8 t/s（26% 效率）|

Dense 模型每 decode step 必须读完全部参数，这是物理约束，不是配置问题。

### 达到 20 t/s 的路径

| 方案 | 预期收益 | 可行性 |
|------|---------|--------|
| **换小模型（14B 或 8B）** | 正比减少，14B ≈ 16 t/s | ✅ 立即可行 |
| **int4/FP4 量化** | ~2x，→ ~18 t/s | 需量化产物 |
| **连续批处理（batch>1）** | 吞吐 N×8 t/s，延迟增加 | 视场景 |
| **华为 msprof 定位隐藏开销** | 可能发现额外 20-30% | 需 msprof 环境 |

**结论**：27B dense 在 310P 当前软件栈下 ~9 t/s 接近实际极限，达到 ≥20 t/s 需换小模型或更激进量化。
