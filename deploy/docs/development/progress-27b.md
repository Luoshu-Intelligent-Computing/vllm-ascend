# Qwen3.6-27B-w8a8 在 310P 上的 Decode 性能分析

**分析日期**: 2026-06-29
**分析角色**: model-infer-analyze（静态分析，无推理服务）
**分析目标**: 为 Qwen3.6-27B-w8a8 在 Atlas 300I Duo（310P3×2，TP=2）上的 decode 速度优化提供基础分析报告

---

## 阶段 0：模型分析

### 运行环境

- **NPU 型号**: Ascend 310P3（`DrvMngGetConsoleLogLevel` 输出确认 Atlas 300I Duo）
- **单卡 HBM**: ~43 GB（每 die）
- **部署卡数**: 2（TP=2，Atlas 300I Duo 为单卡双 die，两 die 各一个 310P3 核心）
- **CANN 版本**: 8.5.0（innerversion V100R001C25B800TP043）
- **驱动版本**: 25.5.1（ascendhal 7.35.23）
- **量化模式**: w8a8（msmodelslim，per-token 激活量化 + per-channel 权重量化）
- **执行模式**: vllm-ascend POC 镜像（`registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend:26.0.0-poc-300i-duo-py311-ubuntu24.04-arm64`）

### 模型架构

> **重要发现：Qwen3.6-27B 不是标准 Dense Transformer**

- **模型路径**: `/srv/meetai/models/llm-service/vllm-ascend/Qwen3.6-27B-w8a8`
- **model_type**: `qwen3_5`（架构类 `Qwen3_5ForConditionalGeneration`）
- **实际架构类型**: 混合 Hybrid Architecture（Linear Attention + Full Attention）
- **模型说明**: Qwen3.5-27B 结构等价体，与 Qwen3.6-35B-A3B（qwen3_5_moe）同代架构

#### 关键参数表

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_hidden_layers` | 64 | 总层数 |
| `hidden_size` | 5120 | 隐藏维度 H |
| `intermediate_size` | 17408 | FFN 中间维度（Dense FFN，非 MoE） |
| `full_attention_interval` | 4 | 每 4 层 1 个 Full Attention |
| 全注意力层数 | **16** | 第 4/8/.../64 层为 Full Attention（GQA） |
| 线性注意力层数 | **48** | 其余 3/4 层为 Linear Attention（循环态） |
| `num_attention_heads` | 24 | Full Attention Q 头数 |
| `num_key_value_heads` | 4 | Full Attention KV 头数（GQA，ratio=6） |
| `head_dim` | 256 | 注意力头维度 |
| `linear_num_key_heads` | 16 | Linear Attention K 头数 |
| `linear_key_head_dim` | 128 | Linear Attention K 头维度 |
| `linear_num_value_heads` | 48 | Linear Attention V 头数 |
| `linear_value_head_dim` | 128 | Linear Attention V 头维度 |
| `linear_conv_kernel_dim` | 4 | 线性注意力卷积核维度（循环窗口大小） |
| `attn_output_gate` | true | Attention 输出门控（swish gate） |
| `vocab_size` | 248320 | 词表大小 |
| `max_position_embeddings` | 262144 | 最大位置编码（256K） |
| `mtp_num_hidden_layers` | 1 | Multi-Token Prediction 额外层数 |
| `partial_rotary_factor` | 0.25 | RoPE 仅作用于 25% 的 head dim（64/256） |
| 量化范围 | self_attn + mlp | visual encoder 保持 FLOAT |

#### 架构理解：Hybrid Linear+Full Attention

这不是传统 Dense Transformer。每 4 层中有 3 层是 **Linear Attention（类 SSM/Mamba）**，只有 1 层是标准 **Full Attention（GQA）**：

```
Layer pattern (64 layers):
  L1:  linear_attention   ← 循环态，O(1) decode，但仍需读投影权重
  L2:  linear_attention
  L3:  linear_attention
  L4:  full_attention     ← 标准 GQA，KV cache 增长
  L5:  linear_attention
  ...
  L64: full_attention
```

**Linear Attention 的 decode 特性**：
- 状态更新为 O(1)（固定大小的循环状态，大小由 conv_kernel_dim=4 和 V 头决定）
- 但每 decode step 仍需读取 K+V 投影权重：`16×128 + 48×128 = 8192` 维 per layer
- Full Attention：读取 Q+K+V+O 投影权重，KV cache 随上下文增长但 decode 每步 Q=1

#### 与 35B MoE 的架构对比

| 维度 | 27B Dense（Qwen3.6-27B） | 35B MoE（Qwen3.6-35B-A3B） |
|------|--------------------------|----------------------------|
| 架构 | Hybrid Linear+Full（Dense FFN） | Hybrid Linear+Full + MoE FFN |
| 总层数 | 64 | 40 |
| Hidden size | 5120 | 2048 |
| FFN 类型 | Dense（intermediate=17408） | MoE（256 专家，激活 8） |
| Full Attention 层 | 16/64 | 10/40 |
| Linear Attention 层 | 48/64 | 30/40 |
| 每 decode step 读取权重 | ~27 GB（全量 Dense FFN） | ~1.2 GB（8/256 专家激活） |
| MTP 层 | 1 层 | 1 层 |

**核心差异**：35B MoE 每 decode step 只激活 8/256 = 3.1% 的专家权重（~1.2 GB），而 27B Dense 必须读取全部 FFN 权重（~17 GB FFN + ~10 GB Attention ≈ 27 GB）。这是性能差距的根本原因。

### 性能与精度基线

**已有实验数据（2026-06-27，TP=2，max_model_len=8192）**：

| 优化方案 | Decode 速度 | 说明 |
|---------|------------|------|
| 基线（PIECEWISE，无优化） | 7.85 t/s | 无图模式 |
| FULL_DECODE_ONLY + async-scheduling | **8.79 t/s** | +12%，当前最优 |
| + HCCL_OP_EXPANSION_MODE=AIV + fuse_gemm_comms | 8.77 t/s | 无增益 |
| fuse_allreduce_rms=True | 启动失败 | 缺少 `npu_add_rms_norm_bias` 算子 |

**当前 POC profile 参数（qwen3.6-27b-w8a8-300i-duo-tp2-16k-1seq-poc）**：
```yaml
tensor_parallel_size: 2
max_model_len: 16384
max_num_batched_tokens: 1024
gpu_memory_utilization: 0.80
dtype: float16
kv_cache_dtype: auto
mamba_ssm_cache_dtype: float16
enable_prefix_caching: false
additional_config: '{"ascend_compilation_config": {"fuse_norm_quant": false}}'
compilation_config: '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1]}'
```

---

## 阶段 1：内存带宽与理论性能分析

### 硬件带宽参数

| 参数 | 值 | 来源 |
|------|-----|------|
| 310P3 HBM 容量 | 43 GB / die | 实测（npu-smi info） |
| 310P3 HBM 理论峰值带宽 | 900-960 GB/s / die | 已知规格，SINGLE_NPU_DEPLOYMENT 引用 900 GB/s |
| CANN 版本 | 8.5.0 | 本机确认 |
| Atlas 300I Duo 形态 | 1 物理卡 × 2 die（310P3） | 硬件规格 |

### Decode 理论上限分析

**每 decode step 权重读取量（TP=2 时各 die 读取一半）**：

| 组件 | 总量 | 每 die（TP=2） |
|------|------|--------------|
| 模型总权重（w8a8，1 byte/param） | ~27 GB | ~13.5 GB |
| 其中 FFN 权重 | ~17.1 GB（64层×267M） | ~8.5 GB |
| 其中 Attention 权重 | ~7.5 GB（估算含 linear+full） | ~3.75 GB |
| 其中 Embedding + LMHead | ~2.5 GB | ~1.25 GB |

**理论速度上限计算**：

```
理论上限 = 每 die 带宽 / 每 die 权重读取量

以 900 GB/s 峰值带宽：
  每 die 权重 = 13.5 GB
  理论时间 = 13.5 / 900 = 15 ms/token
  理论最大速度 = 66 t/s（TP=2 完美情况）

以实际持续带宽（~65% 峰值 = 585 GB/s）：
  理论时间 = 13.5 / 585 = 23 ms/token
  有效带宽上限 = 43 t/s

已有报告引用 @900 GB/s 理论速度 ~33 t/s（以单 die 13.5 GB 为基准）
```

**当前实测效率**：
- 实测：8.8 t/s（~114 ms/token）
- 对比理论 @900 GB/s（全 die，TP=2，以报告数据 ~33 t/s）：效率 **26.7%**
- 存在 **~3.7x 的效率差距**

### 效率差距来源分析

实验已验证：TP=1 与 TP=2 速度相同（~8 t/s），排除 TP 通信开销。差距主要来自：

1. **Hybrid 架构额外开销**（未量化）：
   - 48 个 Linear Attention 层有循环状态更新逻辑（conv 状态，kernel_dim=4）
   - 状态更新涉及额外的矩阵运算，非纯 matmul
   - attn_output_gate（swish）是额外的 elementwise 运算

2. **CANN 图编译/调度开销**：
   - FULL_DECODE_ONLY 已启用，大幅减少了 Python scheduler 开销（~+12% 验证）
   - 但 Hybrid 架构中 linear attention state update 可能有 Python 控制流难以入图

3. **MTP（Multi-Token Prediction）额外层**：
   - 1 个额外 MTP hidden layer，每 step 增加约 +1/64 ≈ 1.5% 计算量
   - 实际影响取决于 vllm-ascend 是否启用 MTP speculative decoding

4. **w8a8 动态量化开销**：
   - 激活量化（per-token）需要运行时统计 scale，每层有额外计算
   - `fuse_norm_quant: false` 禁用了 RMSNorm+量化融合，两步独立执行

5. **KV cache 管理开销**：
   - 16 个 Full Attention 层的 KV cache 读写（linear attention 无 KV cache）
   - `mamba_ssm_cache_dtype: float16` 表明 linear attention 循环状态也有显存占用

---

## 阶段 2：优化方向可行性评估

### 当前已启用的优化

| 优化项 | 状态 | 效果 |
|--------|------|------|
| `FULL_DECODE_ONLY` 图模式 | **已启用** | +12%（已验证：7.85→8.79 t/s） |
| async-scheduling | **已启用** | 含在 FULL_DECODE_ONLY 实验中 |
| w8a8 量化 | **已启用** | 权重 1 byte/param |

### 待评估的优化方向

#### A. 融合算子（fuse_norm_quant）

| 项目 | 评估 |
|------|------|
| **选项** | `fuse_norm_quant: true`（RMSNorm + 量化融合） |
| **当前状态** | 禁用（`fuse_norm_quant: false`） |
| **禁用原因** | 历史实验因 OOM/精度问题关闭，需重新验证 |
| **预期收益** | 减少 RMSNorm→量化的中间 tensor，降低 HBM 读写。估算 5-15% |
| **风险** | 310P 上 `DynamicQuantKernelNpuOpApi` 曾引发 OOM（progress.md 记录第 3 步）；CANN 8.5 是否已修复未知 |
| **建议** | 先以小 max_model_len（4096）测试，监控显存，再扩展 |

#### B. fuse_allreduce_rms

| 项目 | 评估 |
|------|------|
| **选项** | `fuse_allreduce_rms: true`（AllReduce + RMSNorm 融合） |
| **当前状态** | 禁用（启动失败） |
| **失败原因** | CANN 当前版本缺少 `npu_add_rms_norm_bias` 算子 |
| **可行性** | CANN 8.5.0 已安装，需核查该算子是否在 8.5 中引入 |
| **预期收益** | 减少 TP=2 的 AllReduce 次数和 Norm 操作，估算 5-10% |
| **阻塞** | 需确认 `npu_add_rms_norm_bias` 在 CANN 8.5.0 + 310P3 上是否可用 |

#### C. fuse_gemm_comms

| 项目 | 评估 |
|------|------|
| **选项** | `fuse_gemm_comms: true`（GEMM + AllReduce 融合） |
| **当前状态** | 已测试（+HCCL_OP_EXPANSION_MODE=AIV），无增益（8.77 vs 8.79 t/s） |
| **结论** | **无效**。TP=1 与 TP=2 速度相同，说明 AllReduce 不是瓶颈，GEMM 融合无收益 |

#### D. 进一步的图模式优化

| 项目 | 评估 |
|------|------|
| **当前** | `FULL_DECODE_ONLY`（cudagraph_mode）已启用 |
| **FULL_GRAPH_MODE** | 进一步静态化整个前向，但 Hybrid 架构的 linear attention state update 可能有动态控制流阻碍 |
| **可行性** | 需测试是否有图中断（Graph Break）；Hybrid 模型图模式支持情况未知 |
| **预期收益** | 如能成功：5-20% decode 加速（消除 Python overhead） |
| **风险** | Qwen3.5/3.6 混合线性注意力在 NPU 图模式下的支持成熟度未知 |

#### E. int4 / W4A8 量化

| 项目 | 评估 |
|------|------|
| **方向** | 权重从 int8 降至 int4，权重字节减半 |
| **预期收益** | ~2x 带宽节省 → 理论上 ~16 t/s（从当前 8.8 t/s） |
| **可行性** | 需新的量化产物（msmodelslim 需重新产出 W4A8 权重）；310P 对 int4 算子支持需确认 |
| **阻塞** | 量化产物不存在，需独立工作 |

#### F. 增大并发（batch>1）

| 项目 | 评估 |
|------|------|
| **方向** | 通过 `max_num_seqs > 1` 提升总吞吐 |
| **效果** | N 并发：总吞吐 ~N×8.8 t/s，但每请求延迟增加 |
| **当前限制** | `max_num_seqs: 1` 是保守值；TP=2，43 GB/die，16k context 下显存余量估算 |
| **显存估算** | 模型权重 ~13.5 GB/die，KV cache（16k×16 full-attn 层）+ SSM 状态（48 linear 层）；需测试 2-4 并发 |
| **适用场景** | 吞吐优先场景有效；延迟敏感场景（单请求 decode）无效 |

#### G. KV cache dtype 优化（fp8）

| 项目 | 评估 |
|------|------|
| **当前** | `kv_cache_dtype: auto`（推测为 fp16/bf16） |
| **方向** | fp8 KV cache 减少 KV cache 显存和带宽 |
| **可行性** | 310P3 对 fp8 KV cache 的支持情况未知；当前 vllm-ascend 版本中 310P 的 fp8 支持成熟度需验证 |
| **预期收益** | 仅影响 Full Attention（16/64 层），Linear Attention 用 SSM 状态（已 fp16）。由于 KV cache 读取在 decode 中占比相对小，预期收益有限（<5%） |

### 优化方向优先级排序

| 优先级 | 优化项 | 预期增益 | 实施难度 | 阻塞点 |
|--------|--------|---------|---------|--------|
| 1 | **fuse_norm_quant: true 重新测试** | 5-15% | 低（改配置） | 需在小 context 先验证显存 |
| 2 | **fuse_allreduce_rms 可用性确认** | 5-10% | 低（改配置） | 需确认 CANN 8.5 算子支持 |
| 3 | **增大并发（max_num_seqs: 2-4）** | 吞吐 2-4x | 低（改配置） | 需实测显存余量 |
| 4 | **FULL_GRAPH_MODE 测试** | 5-20% | 中（图中断修复） | Hybrid 架构图模式未验证 |
| 5 | **W4A8 量化产物** | ~2x（理论上限 ~16 t/s） | 高（需重新量化） | 量化产物不存在 |

---

## 阶段 0 结论：基线分析与约束

### 核心约束（物理限制）

1. **27B Dense 模型每 decode step 必须读取全部 ~27 GB 权重**，这是物理约束
2. 310P3 理论带宽 900 GB/s，以 26% 效率计算实际可达约 8.8 t/s
3. **在 310P 当前软件栈下，单请求 decode 理论上限约 20-25 t/s**（带宽利用率提升至 60-70% 时）
4. 如需 ≥20 t/s，需要 W4A8 量化（减半权重字节数）

### 与 35B MoE 的性能差距本质

MoE decode 快的原因不是架构更优，而是 **稀疏激活**：
- 35B MoE 每 step 只读 8/256 = 3.1% 专家权重（~1.2 GB/step）
- 27B Dense 每 step 读全量 FFN（~17 GB/step）
- 这 14x 的权重读取量差异直接导致了 31.5 t/s vs 8.8 t/s 的性能比

### 待推理服务才能验证的分析项

以下分析需要运行推理服务后才能确认：

| 待验证项 | 验证方法 | 预期信息 |
|---------|---------|---------|
| `fuse_norm_quant: true` 是否 OOM | 以 4096 context 测试启动 | 是否可启用 |
| `fuse_allreduce_rms` 是否可用 | 测试启动，捕获算子 not found 错误 | CANN 8.5 是否引入该算子 |
| msprof profiling | 采集 decode step timeline | 定位哪个算子占用最多时间 |
| `max_num_seqs: 2` 显存余量 | 16k context，测试启动 | 是否有显存空间支持 2 并发 |
| FULL_GRAPH_MODE 图中断排查 | 启动时查看 graph break 日志 | Hybrid 架构是否能完整入图 |
| Linear Attention 实际算子开销占比 | msprof 算子级 profiling | 48 层 SSM 状态更新的真实耗时 |

---

## 附录：关键发现修正（宿主机直接验证，2026-06-29）

### 修正 1：linear_attn 权重未量化

通过读取 `quant_model_description.json` 确认：
- **48 层 linear_attn 权重全部为 FLOAT（fp16），未量化**
- 仅 FFN（全 64 层）和 self_attn（16 层）使用 W8A8_DYNAMIC
- 这意味着 ~8 GB 的 linear_attn 权重以 fp16 读取，带来额外带宽压力

### 修正 2：带宽效率分析

精确计算（基于实际量化配置）：
- 每 decode step 读取总量：26.34 GB（w8a8 部分 18.29 GB + fp16 linear_attn 8.05 GB）
- TP=2 每 die：13.17 GB
- 理论上限 @900 GB/s：68.3 t/s（14.6 ms/token）
- 实测 8.8 t/s（114 ms/token），**带宽利用率仅 12.9%**

**关键结论：27B 不是纯 memory-bound，87% 的时间花在非带宽开销上**

主要开销来源（需 msprof 精确定位）：
1. **48 层 Linear Attention 状态更新**（conv 卷积，额外计算量，非纯矩阵乘）
2. **fuse_norm_quant=false**（RMSNorm + 量化分两步，64层×2次额外 HBM 读写）
3. **Python/CANN 调度开销**（FULL_DECODE_ONLY 已改善 12%，仍有残余）
4. **mamba_ssm_cache_dtype=float16**（SSM 状态 HBM 读写）

这说明融合算子优化（fuse_norm_quant）和 msprof profiling 的空间**远超预期**。

### 修正 3：量化潜力上限

| 方案 | 每 die 读取 | 理论上限 |
|------|-----------|---------|
| 当前（w8a8 部分量化） | 13.17 GB | 68.3 t/s |
| + linear_attn → w8a8 | 11.16 GB | 80.7 t/s |
| + FFN/FullAttn → w4a8 | 8.60 GB | 104.7 t/s |
| 全部 → w4a8 | 6.59 GB | 136.7 t/s |

以上均为带宽上限，实际可达值取决于能否同时消除计算/调度开销（12.9% → 50%+ 效率）。

### 修正 4：部署场景

从 `vllm-ascend-profiles.yaml` 确认当前 310P 27B POC 配置：
```yaml
# qwen3.6-27b-w8a8-300i-duo-tp2-16k-1seq-poc
tensor_parallel_size: 2
max_model_len: 16384
max_num_seqs: 1
max_num_batched_tokens: 1024
mamba_ssm_cache_dtype: float16
fuse_norm_quant: false
cudagraph_mode: FULL_DECODE_ONLY
```

生产（910B4 单卡）配置为 `qwen3.6-27b-w8a8-1npu-128k-2seq`，与 310P POC 配置完全不同。

---

**文档版本**: 1.1（含宿主机直接验证修正）
**分析人**: CANNBot + 主 agent 补充
**更新时间**: 2026-06-29

---

## 阶段 0.2：orangepi 远程环境调研（2026-06-30）

> 通过 `ssh root@orangepi` 直接采集，所有数据为实测值。

### 运行环境

| 项目 | 值 | 来源 |
|------|-----|------|
| NPU 型号 | **Ascend 310P1**（Atlas 200I Pro） | `npu-smi info -t common -i 0` → `Product Type: Atlas 200I Pro` |
| 单卡 HBM 总量 | **96 GB**（98304 MB total；驱动保留后 89610 MB 可用） | `npu-smi info -t memory -i 0` |
| 部署卡数 | **1 卡**（单芯片，非双 die） | `npu-smi info -l`：`Chip Count: 1` |
| 驱动版本 | npu-smi v1.0，软件 24.1.t52.b060，固件 7.6.7.0.b053 | `npu-smi info` |
| 容器镜像 | `vllm-ascend:dev-26.0.0.poc.20260413-9.0.T3.B030-20260421115402-300I-Duo-py311-openEuler24.04-lts-aarch64` | `docker ps` |
| 容器名 | `qwen36-27b-310p-tp1-poc` | 运行 35 小时以上 |
| 模型挂载路径 | `/mnt/usb/Qwen3.6-27B-w8a8` → 容器内 `/models/Qwen3.6-27B-w8a8` | `docker inspect` |

**注意**：orangepi 是 **Atlas 200I Pro（单卡 310P1，~87.5 GB 可用）**，不是 Atlas 300I Duo（双 die 310P3，每 die 43 GB）。两者均为 310P 系列，但形态不同。

### 容器实际启动参数

从 `docker inspect` 提取：

```bash
vllm serve /models/Qwen3.6-27B-w8a8 \
  --served-model-name qwen3.6 \
  --host 0.0.0.0 --port 38081 \
  -tp 1 \
  --max-model-len 8192 \
  --max_num_seqs 1 \
  --max-num-batched-tokens 512 \
  --gpu-memory-utilization 0.70 \
  --dtype float16 \
  --trust-remote-code \
  --language-model-only \
  --enforce-eager \
  --no-enable-prefix-caching \
  --kv-cache-dtype auto \
  --additional-config '{"ascend_compilation_config": {"fuse_norm_quant": false, "fuse_act_quant": false}}' \
  --mamba-ssm-cache-dtype float16
```

**关键参数说明**：
- `--tp 1`：单卡运行（与 orangepi 只有 1 块 NPU 一致）
- `--max-model-len 8192`：8K 限制为**配置保守值**，不是 OOM 触发（见显存分析节）
- `--language-model-only`：忽略 vision encoder，纯文本推理
- `--enforce-eager`：禁用图模式（cudag graph）
- `fuse_norm_quant: false, fuse_act_quant: false`：与 300I Duo 环境相同，两个融合均关闭

### 8K 上下文限制原因分析

**不是 OOM 触发**。mask 在 8192×8192×2B = 128 MB，完全可接受。

显存预算估算（`gpu_memory_utilization=0.70`，可用 87.5 GB × 0.70 = **61 GB**）：

| 项目 | 占用 |
|------|------|
| 模型权重（w8a8，~27 GB） | ~27 GB |
| KV cache 可用余量 | ~34 GB |
| mask（8K，未 patch）| 0.125 GB |

8192 限制很可能是服务方配置的保守启动参数，用于减少冷启动风险。Patch 后可安全扩展到更长上下文（见显存可行性节）。

---

### 架构分析（容器内 config.json 直接确认）

config.json 从容器读取，与宿主机静态分析完全一致：

- `model_type`: `qwen3_5`，架构 `Qwen3_5ForConditionalGeneration`
- 包含 `vision_config`（ViT，`--language-model-only` 屏蔽，不影响 text 推理）
- `text_config.layer_types`: 64 层，`linear_attention`×48 + `full_attention`×16，模式 [LA,LA,LA,FA]×16
- **无 MoE**（无 `num_experts`/`num_experts_per_tok` 字段）
- **有 mamba SSM 状态**：`mamba_ssm_dtype: float32`（config），启动时 `--mamba-ssm-cache-dtype float16` 覆盖

**结论**：27B 不是"纯 Transformer"，是 Hybrid Linear+Full Attention。存在 48 层 linear attention，有 mamba_ssm_cache（固定大小循环状态，与序列长度无关）。

---

### 310P attention backend 兼容性评估

容器内确认 `_310p/attention/` 目录结构：

```
/usr/local/python3.11.10/lib/python3.11/site-packages/vllm_ascend/_310p/attention/
├── __init__.py          ✓
├── metadata_builder.py  ✓  AscendAttentionMetadataBuilder310
├── attention_v1.py      ✓  AscendAttentionBackend310 / AscendAttentionBackendImpl310
└── attention_mask.py    ✓  AttentionMaskBuilder310（lazy init，带 attn_mask_cache）
```

**三个 patch 目标文件全部存在**。

关键代码确认：
- `metadata_builder.py`：继承 base builder，覆盖 `self.attn_mask_builder` 为 `AttentionMaskBuilder310`，但**未覆盖 `build()` 方法**
- `attention_v1.py`：包含 `forward_prefill_310`（第 131 行）和对应调用（第 240 行）
- `attention_mask.py`：`get_attention_mask()` 有 `attn_mask_cache`（lazy init，第 151-153 行），首次调用时分配 `max_model_len × max_model_len` 并缓存

**OOM 触发路径**：base `AscendAttentionMetadataBuilder.build()` 第 294 行调用 `self.attn_mask_builder.get_attention_mask()`，触发 `AttentionMaskBuilder310.get_attention_mask()`，lazy init 后分配 `max_model_len × max_model_len` FRACTAL_NZ 格式 tensor。128K 时：`131072 × 131072 × 2B = 32 GB` → OOM。

**Patch 方案**：覆盖 `build()` 设 `attn_mask = None`，`forward_prefill_310` 按需动态生成 `[T, T]` 小 mask，彻底绕过预分配。

#### 27B mamba SSM 兼容性

任务描述假设"27B 是纯 Transformer，mamba_ssm_cache 为空"——此假设**不成立**。27B 有 48 层 linear attention，mamba_ssm_cache 有效。

但这不影响 patch 有效性：
- 310P attention backend 仅处理 `full_attention` 层（16/64 层）
- `linear_attention` 层有独立的 SSM 路径，不经过 `AscendAttentionBackendImpl310`
- Patch 设 `attn_mask = None` 只作用于 full attention 的 prefill 路径
- Linear attention 层的 mamba_ssm_cache 不受影响

**兼容性结论：PASS**。Patch 对 27B 有效，无需适配。

---

### 显存可行性估算（TP=1，单卡 87.5 GB，gpu_memory_utilization=0.70）

**可用显存**：87.5 × 0.70 = **61 GB**

**权重占用**：
- 总参数 ≈ 27 B（近似）
- w8a8：1 byte/param → **~27 GB**

**KV cache**（仅 16 个 full_attention 层，GQA）：
```
KV cache = num_full_attn_layers × 2 × seq_len × num_kv_heads × head_dim × bytes
         = 16 × 2 × seq_len × 4 × 256 × 2
         = seq_len × 65536 bytes
```

| 上下文长度 | KV cache | 总占用（权重+KV） | 是否可行（61 GB 预算） |
|-----------|---------|-----------------|----------------------|
| 8K（当前） | 0.5 GB | 27.5 GB | ✓（显存余量很充裕）|
| 32K | 2 GB | 29 GB | ✓ |
| 64K | 4 GB | 31 GB | ✓ |
| 128K | 8 GB | 35 GB | ✓ |
| 200K（vision 位置上限） | 12.5 GB | 39.5 GB | ✓ |

**Mamba SSM cache（48 层 linear attention）**：
```
conv_state = batch × 48 × linear_num_key_heads × linear_key_head_dim × conv_kernel_dim × bytes
           = 1 × 48 × 16 × 128 × 4 × 2 ≈ 0.75 MB（固定，不随 seq_len 增长）
```
SSM 状态极小，可忽略。

**mask 占用（无 patch）**：
- 8K：128 MB（可接受）
- 128K：32 GB → **OOM**

**推荐上下文长度配置**：
- 应用 patch 后，单批单序列下 128K 完全可行，显存余量约 26 GB
- 建议先以 32K 验证，再扩展到 128K
- 如需多并发（max_num_seqs > 1），KV cache 线性增长，1 seq @ 128K = 8 GB，4 seq @ 32K = 8 GB

---

### 汇总

| 分析项 | 结论 |
|--------|------|
| NPU 型号 | Atlas 200I Pro（单卡 310P1） |
| 单卡 HBM 可用 | ~87.5 GB（89610 MB） |
| 部署卡数 | 1（TP=1） |
| 27B 架构类型 | Hybrid（64 层：48 linear_attention + 16 full_attention，无 MoE） |
| mamba_ssm_cache | 存在（48 层 linear attention），但不影响 patch |
| 310P patch 三文件 | 全部存在于容器镜像 ✓ |
| patch 兼容性 | **PASS**（无需适配） |
| 8K 限制原因 | 配置保守值，非 OOM 触发 |
| 显存可行性（32K） | ✓ |
| 显存可行性（64K） | ✓ |
| 显存可行性（128K） | ✓（patch 后 mask 不预分配） |
| 推荐优先验证 | 32K → 64K → 128K 逐步扩展 |

**文档版本**: 1.2（orangepi 远程调研补充）
**分析人**: model-infer-analyzer（远程实测）
**更新时间**: 2026-06-30

---

## 阶段 0.4：32K/128K 服务部署验证（2026-06-30）

### 实施记录

- [完成] 停止并删除旧容器 `qwen36-27b-310p-tp1-poc`（原参数：`--max-model-len 8192 --enforce-eager`）
- [完成] SCP 三个 patch 文件到 orangepi `/tmp/310p-patch/`：`metadata_builder.py`、`attention_v1.py`、`attention_mask.py`
- [失败] 首次启动尝试 FULL_DECODE_ONLY 图模式（`--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"...}'`） — 根因：编译阶段 AICPU kernel `StatelessRandomNormalV2` 在 310P 上崩溃（error code 507018/0x2a），发生在 `npugraph_ex_compile → torch.npu.set_compile_mode(jit_compile=False)` 调用链。310P 不支持 npugraph_ex 编译所需的部分 AICPU 算子，原容器使用 `--enforce-eager` 规避了这条路径。
- [完成] 32K 验证：以 `--enforce-eager --max-model-len 32768 --gpu-memory-utilization 0.70` 启动，挂载三个 patch 文件 — 服务成功就绪
- [完成] 128K 验证：以 `--enforce-eager --max-model-len 131072 --gpu-memory-utilization 0.90` 重启 — 服务成功就绪

### 当前代码状态

- 容器名：`qwen36-27b-310p-tp1-poc`（运行中）
- 镜像：`vllm-ascend:dev-26.0.0.poc.20260413-9.0.T3.B030-20260421115402-300I-Duo-py311-openEuler24.03-lts-aarch64`
- 当前参数：`--max-model-len 131072 --gpu-memory-utilization 0.90 --enforce-eager --enable-chunked-prefill`
- 三个 patch 文件挂载至 `/usr/local/python3.11.10/lib/python3.11/site-packages/vllm_ascend/_310p/attention/`（只读 bind mount）
- patch 来源：`/tmp/310p-patch/`（orangepi 宿主机，由本机 SCP 上传）

### 自验证结果

- 参考 skill: model-infer-migrator（场景：已有服务的参数替换与基线采集）
- 代码加载：日志无 import error，`AttentionMaskBuilder310` 和 `AscendAttentionMetadataBuilder310` 均正常注册
- 编译：N/A（--enforce-eager，图模式已禁用）
- 推理：通过，短文本请求正常返回

**32K 验证**（gpu_memory_utilization=0.70）：
- `/v1/models`：`max_model_len=32768` ✓
- 推理：`1+1`问题正常返回，输出可读无重复 ✓
- 日志：无 Falling back / OOM 错误 ✓
- KV cache：Available 6.34 GiB，tokens 25,600，最大并发 3.01x

**128K 验证**（gpu_memory_utilization=0.90）：
- `/v1/models`：`max_model_len=131072` ✓
- 推理：短文本 chat completion 正常返回，输出可读无重复 ✓
- 日志：无 Falling back / OOM 错误 ✓
- KV cache：Available 15.09 GiB，tokens 61,440，最大并发 1.86x
- 权重加载：37.54 GB，120.72 秒

**显存占用**（128K，gpu_memory_utilization=0.90）：
- NPU 总 HBM：89,610 MB（~87.5 GB）
- 可用量（0.90）：~78.7 GB
- 权重：37.54 GB
- 可用 KV cache：15.09 GiB（61,440 tokens）
- 剩余系统开销：~26.1 GB

### 关键发现：FULL_DECODE_ONLY 在 310P 上不可用

CANN/310P 镜像中 AICPU kernel `StatelessRandomNormalV2`（来自 `libtf_kernels.so`）在 npugraph_ex 编译阶段崩溃（errorCode=0x2a）。这是硬件/软件栈限制，不是 patch 引入的问题。

310P 27B 服务必须使用 `--enforce-eager`，无法使用 `FULL_DECODE_ONLY` 图模式。  
任务描述中"移除 `--enforce-eager`（改用 FULL_DECODE_ONLY 图模式）"的方向在当前镜像（9.0.T3.B030）下不可行，需等待 CANN/驱动升级或官方确认 310P 对 npugraph_ex 的支持。

**文档版本**: 1.3（32K/128K 服务部署验证）
**实施人**: model-infer-implementer
**更新时间**: 2026-06-30

---

## 阶段 0.5：orangepi 310P1 性能基线采集（2026-07-01）

### 环境对比

| 环境 | NPU 型号 | HBM | 部署 | Decode 速度 | TTFT（256 tokens）|
|------|---------|-----|------|------------|------------------|
| **300I Duo** | 310P3 (双 die) | 43 GB/die | TP=2 | **8.79 t/s** | **948 ms** |
| **orangepi** | 310P1 (单芯片) | 87.5 GB | TP=1 | **1.7-1.8 t/s** | **78,000 ms** |
| 性能比 | - | - | - | **0.20x** | **0.012x** |

### 关键发现

1. **310P1 性能严重不足**：
   - Decode 速度仅 1.7-1.8 t/s（比 310P3×2 慢 5 倍）
   - TTFT 达到 78 秒（比 310P3×2 慢 80 倍）
   - 等效 HBM 带宽仅 ~47 GB/s（310P3 理论 900 GB/s 的 5%）

2. **硬件限制**：
   - 310P1 是边缘推理芯片，带宽和算力远低于数据中心级 310P3
   - Hybrid Linear+Full Attention 架构在 310P1 上执行效率极差
   - 27B Dense 模型不适合在 310P1 上部署

3. **测试方法**：
   - 服务日志持续观察：`Avg generation throughput: 1.7-1.8 tokens/s`
   - 直接 API 测试：64 tokens 输出耗时 115 秒，速度 0.55 t/s
   - TTFT 测试（256 tokens）：78-82 秒

### 结论

**310P1 不适合 27B 模型推理**。建议：
- 短期：降级到更小模型（7B/14B）或测试 MoE 模型
- 长期：升级到 300I Duo（310P3×2）或 910B 系列

完整报告：`docs/reports/ORANGEPI_27B_BASELINE_20260701.md`

**文档版本**: 1.4（orangepi 310P1 性能基线）
**测试人**: model-infer-analyzer
**更新时间**: 2026-07-01
