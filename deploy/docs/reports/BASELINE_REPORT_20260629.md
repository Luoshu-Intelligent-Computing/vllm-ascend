# Qwen3.6-35B-A3B-w8a8 @ 310P 基线报告

**模型**: Qwen3.6-35B-A3B-w8a8（MoE，3.5B 激活参数，w8a8 量化）
**硬件**: Atlas 300I Duo (310P3 × 2，共 ~87 GB HBM)
**框架**: vllm-ascend 26.0.0
**报告日期**: 2026-06-29
**状态**: ✅ 生产可用（128K 已验证）

---

## 服务配置（基准）

```bash
--max-model-len 131072             # 128K 上下文
--max-num-batched-tokens 1024      # ⚠️ 310P 安全上限，不得调高
--tensor-parallel-size 2
--gpu-memory-utilization 0.75
--enable-chunked-prefill
--no-enable-prefix-caching
--reasoning-parser qwen3
--mamba-ssm-cache-dtype float16
--additional-config '{"ascend_compilation_config": {"fuse_norm_quant": false}}'
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1]}'
```

---

## 一、显存基线

| 配置 | NPU 卡0 | NPU 卡1 | 利用率 |
|------|--------|--------|--------|
| 128K 服务（空闲）| 32,702 MB / 44,278 MB | 32,198 MB / 43,693 MB | ~73.8% |
| 64K 服务（空闲）| 32,178 MB / 44,278 MB | 31,675 MB / 43,693 MB | ~72.6% |

Patch 节省的显存：mask 从 8 GB（原方案）→ 8 MB（动态生成），节省 -99.9%。

---

## 二、性能基线

> 测试方法：流式 API，3 次重复取中位数，enable_thinking=False

### 2.1 首字延迟（TTFT）& E2E 延迟

| 输入规模（实际 tokens）| TTFT (ms) | E2E 延迟（+128 out，ms）|
|:--------------------:|:---------:|:---------------------:|
| 256（301t） | 948 | 1,935 |
| 1K（1,197t）| 2,839 | 3,858 |
| 4K（4,781t）| 8,531 | 9,529 |
| 8K（9,555t）| 15,094 | 16,103 |
| 16K（19,117t）| 31,542 | 32,479 |
| 32K（38,227t）| 64,520 | 65,496 |

### 2.2 Prefill 吞吐

| 输入规模 | Prefill 吞吐 |
|---------|------------|
| 256t | 318 t/s |
| 1K t | 422 t/s |
| 4K t | 560 t/s |
| 8K t（峰值）| **633 t/s** |
| 16K t | 606 t/s |
| 32K t | 593 t/s |

> ⚠️ 与旧参数（max_num_batched_tokens=2048，峰值~960 t/s）相比下降约 34%，
> 这是 310P ATB 算子精度修复的必要代价。

### 2.3 Decode 速度（纯解码，与输出长度无关）

| 输出长度 | Decode 速度 | ms/token |
|---------|------------|---------|
| 32t | 32.9 t/s | 30.4 ms |
| 64t | 32.2 t/s | 31.1 ms |
| 128t | 31.4 t/s | 31.8 ms |
| 256t | 31.6 t/s | 31.6 ms |
| 512t | 31.6 t/s | 31.7 ms |

**Decode 速度：~31.5 t/s，~31.7 ms/token（极稳定，与输出长度无关）**

---

## 三、精度基线

### 3.1 GSM8K（数学推理，短上下文）

| 服务 | 样本数 | 准确率 | max_tokens | 说明 |
|------|:-----:|:-----:|:---------:|------|
| 64K 服务 | 50 | **98.0%** | 2000 | Phase 4 基线 |
| 128K 服务 | 50 | **98.0%** | 2000 | 与 64K 一致 |

唯一错误（题目 #12）：pred=12.0 vs correct=13.0，数学推理 off-by-1，非框架问题。

### 3.2 GPQA-Diamond（博士级 MCQ，深度推理）

| 样本数 | 准确率 | max_out_len | thinking | 说明 |
|:-----:|:-----:|:-----------:|:-------:|------|
| 5 | 100% | 15000 | ✅ 开启 | 冒泡测试 |
| **20** | **80.0%** | 15000 | ✅ 开启 | 正式样本 |

GPQA-Diamond 参考分数对比：

| 模型 | GPQA-Diamond |
|------|:-----------:|
| GPT-4o | ~53% |
| Claude 3.5 Sonnet | ~65% |
| Qwen3-32B（稠密）| ~65-70% |
| **本模型（20样本）** | **80%** |

> ⚠️ 20 样本统计误差较大（置信区间约 ±20%），建议后续全量 198 题确认。
> 模型输出长度 8K-26K chars/题，thinking 模式正常。

---

## 四、关键参数约束汇总

| 参数 | 推荐值 | 禁止值 | 原因 |
|------|:-----:|:-----:|------|
| `max_num_batched_tokens` | **1024** | ≥2048 | ATB 算子静默精度损坏 |
| `max_model_len` | 65536 或 131072 | >131072 | 显存限制 |
| `enable_chunked_prefill` | **必须开启** | 关闭 | 长上下文必需 |
| `mamba_ssm_cache_dtype` | float16 | - | 当前配置 |
| thinking 模式 | 按场景选择 | - | 深度推理开启，快速问答关闭 |

---

## 五、历史版本对比

| 指标 | v0（原始）| v1（64K patch）| v2（当前 128K）|
|------|:--------:|:-------------:|:-------------:|
| 最大上下文 | 32768 | 65536 | **131072** |
| 显存/卡（空闲）| ~36 GB | ~32 GB | ~31.7 GB |
| mask 显存 | 8 GB（永久）| 8 MB（动态）| 8 MB（动态）|
| Prefill 峰值 | - | ~633 t/s | ~633 t/s |
| Decode 速度 | - | ~31.5 t/s | ~31.5 t/s |
| GSM8K | - | 98% | 98% |
| GPQA-Diamond | - | - | **80%（20样本）** |
| 长上下文精度 | ❌ | ✅ | ✅ |

---

## 六、测试文件索引

| 文件 | 说明 |
|------|------|
| `docs/reports/PERFORMANCE_BASELINE_20260629.md` | 详细性能数据（TTFT/E2E/Decode） |
| `docs/reports/GPQA_DIAMOND_20260629.md` | GPQA-Diamond 20 样本结果 |
| `docs/reports/LONGBENCH_AISBENCH_20260629.md` | LongBench dureader 200 样本结果 |
| `docs/reports/LONGBENCH_EVALUATION_20260629.md` | LongBench 手动脚本验证报告 |
| `docs/reports/GSM8K_COMPARISON_20260626.md` | GSM8K 64K vs 128K 对比 |
| `docs/guides/DEPLOYMENT_GUIDE.md` | 生产部署手册 |

---

**维护人**: CANNBot model-infer-optimize
**日期**: 2026-06-29
