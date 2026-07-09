# 310P 128K 服务并发能力分析与优化方向

**记录时间**: 2026-07-08  
**硬件**: Atlas 300I Duo (310P3 × 2，共 ~87 GB HBM)  
**模型**: Qwen3.6-35B-A3B-w8a8  
**镜像**: `registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend:310p-opt-20260708`

---

## 一、当前服务配置（起点）

```bash
--max-model-len 131072          # 128K 上下文
--max-num-seqs 8                # 最大并发请求数
--max-num-batched-tokens 1024   # ⚠️ 310P ATB 精度约束上限，不可调高
--gpu-memory-utilization 0.75
--kv-cache-memory 21653924864   # 20.17 GiB（优化后）
--enable-chunked-prefill
```

**KV Cache 分配**：

| 配置项 | 当前值 |
|--------|--------|
| KV cache memory | 20.17 GiB（可容纳 ~2M tokens） |
| 权重占用 | 18.86 GiB / 卡 |
| 峰值激活 | ~1.7 GiB |
| 有效 KV token 容量 | ~473K tokens（原始 4.59 GiB 时；优化后约 2M） |

---

## 二、8 并发实测结果（2026-07-08）

### 测试配置

- 并发数：8
- 测试类型：混合短任务（简短问答、翻译、编程、数学等）
- 输入 tokens：~20/请求
- 目标输出：30-100 tokens/请求

### 实测数据

| ID | 任务 | 延迟(ms) | 输入 | 输出 |
|----|------|---------|------|------|
| 1 | 简短问答 | 15,660 | 17 | 50 |
| 2 | 计算题 | 23,118 | 31 | 80 |
| 3 | 写作 | 7,859 | 17 | 24 |
| 4 | 常识 | 18,183 | 19 | 60 |
| 5 | 编程 | 27,911 | 20 | 100 |
| 6 | 翻译 | 10,102 | 19 | 30 |
| 7 | 历史 | 15,655 | 17 | 50 |
| 8 | 数学 | 27,908 | 20 | 100 |

**汇总指标**：

| 指标 | 数值 |
|------|------|
| 成功率 | **8/8（100%）** |
| 总耗时 | 27,917 ms |
| 平均延迟 | 18,299 ms |
| P50 延迟 | 16,922 ms |
| P95 延迟 | 27,911 ms |
| 系统总吞吐 | **17.7 t/s**（所有请求合计输出） |
| 总输入 tokens | 160 |
| 总输出 tokens | 494 |

### 与单并发对比

| 指标 | 单并发（基线）| 8并发（实测）| 理论8并发上限 |
|------|:-----------:|:-----------:|:------------:|
| 单请求 decode 速率 | 31.5 t/s | ~3.6 t/s | ~31 t/s |
| 系统总吞吐 | 31.5 t/s | **17.7 t/s** | ~252 t/s |
| 100 tokens 端到端延迟 | ~3.2 s | ~27.9 s | ~3.5 s |

---

## 三、根因分析

### 实际吞吐远低于理论值的原因

理论上 8 并发应能达到 ~252 t/s 总吞吐（8 × 31.5 t/s），但实测仅 17.7 t/s，差距约 14×。

**根本原因：`max_num_batched_tokens=1024` 导致 prefill-decode 相互阻塞**

```
时间轴（示意）：
[Prefill chunk 1/请求A][Prefill chunk 2/请求A]...[Decode step 1][Decode step 2]...
                         ← prefill 阻塞 decode → pipeline bubble 严重
```

- Chunked prefill 每步只处理 1024 tokens
- 8 个请求的 prefill 轮流占用处理器
- 频繁打断 decode 流程，产生大量 pipeline bubble
- 实际 decode 密度（有效 decode 步骤占总时间比例）极低

### 为什么不能提高 max_num_batched_tokens

ATB 算子（`_npu_paged_attention_splitfuse`）存在精度约束：

| batch_tokens | 精度状态 |
|:----:|:------:|
| ≤ 1024 | ✅ 正常（已验证）|
| 2048 | ❌ 静默精度损坏（实测 LongBench 全错）|
| 1536 | ⚠️ 未测试 |

---

## 四、并发能力估算（理论）

基于 KV cache 容量（~20 GiB = 2M tokens），不同场景最大并发数：

| 场景 | 平均上下文 | 最大并发（理论）| 推荐并发 |
|------|:--------:|:-----------:|:-------:|
| 短对话（2K tokens）| 2,048 | ~1,000 | 8-16 |
| 中等对话（8K tokens）| 8,192 | ~250 | 4-8 |
| 长文档（32K tokens）| 32,768 | ~62 | 2-4 |
| 超长上下文（128K tokens）| 131,072 | ~15 | 1-2 |

> ⚠️ 上述理论值受 KV cache 容量限制，实际可用并发还受 decode 吞吐和 chunked prefill 调度约束。

---

## 五、优化方向

### 方案 A：调整 max-num-seqs（立即可用，无风险）

针对不同业务场景选择合适的并发数，权衡延迟和系统利用率：

**实时交互场景**（用户等待响应，延迟敏感）：
```bash
--max-num-seqs 2-3
# 预期延迟：~8-12s（100 tokens 输出）
# 系统吞吐：~25-28 t/s
```

**混合场景**（部分交互+部分批处理）：
```bash
--max-num-seqs 4-6
# 预期延迟：~15-20s（100 tokens 输出）
# 系统吞吐：~20-25 t/s
```

**批处理场景**（异步任务，不关注延迟）：
```bash
--max-num-seqs 8-16
# 预期延迟：~30-60s（100 tokens 输出）
# 系统吞吐：~17-22 t/s
```

### 方案 B：实验性测试 max_num_batched_tokens=1536（需精度验证）

`2048` 已确认精度损坏，中间值 `1536` 未验证：

```bash
# 步骤 1：用 1536 启动测试服务
--max-num-batched-tokens 1536

# 步骤 2：精度回归测试
python3 gsm8k_evaluation.py 50     # 目标 ≥96%
python3 test_128k_boundary.py      # 长上下文精度

# 步骤 3：若通过，性能基准测试
python3 perf_benchmark_detailed.py
```

**预期收益**（若精度通过）：
- Chunked prefill 块更大 → pipeline bubble 减少
- 预计系统吞吐从 17.7 → 25-30 t/s
- 并发延迟从 ~28s → ~18-20s

**风险**：可能存在精度损坏（需验证后才能使用）。

### 方案 C：按业务场景分离部署

| 服务实例 | 端口 | max-num-seqs | 适用场景 |
|---------|------|:------------:|---------|
| 实时对话服务 | 18082 | 3 | 前端交互、即时响应 |
| 批处理服务 | 18083 | 8 | 后端任务、异步处理 |

### 方案 D：等待上游 ATB 算子修复（长期）

vllm-ascend 官方已知此问题，后续 CANN 商用版中 `_npu_flash_attention_v3` + `_npu_paged_attention_splitfuse_v2` 支持后：
- compressed mask 方案替换 chunked splitfuse mask
- 消除 `max_num_batched_tokens` 精度约束
- 理论上可恢复 ~960 t/s prefill 吞吐（当前 633 t/s）

---

## 六、推荐行动

**近期（可立即执行）**：

1. **确定业务场景**：交互优先还是吞吐优先？
2. **调整 max-num-seqs**：
   - 交互优先 → 3-4 并发，延迟 ~10-15s
   - 吞吐优先 → 8 并发，延迟 ~28s
3. **测试 max_num_batched_tokens=1536**（低风险，高收益）

**中期**：

4. **压测不同场景**：用真实业务请求（而非均匀短任务）测试
5. **监控 KV cache 使用率**：确认 20 GiB 分配是否充足

**长期**：

6. **跟踪上游 ATB 算子更新**：等待 compressed mask v3/v2 算子商用化
7. **考虑 causal_fa_310p hang 修复**：修复后可恢复 960 t/s prefill 基线

---

## 七、测试脚本

并发测试脚本位于：`deploy/tests/perf_benchmark_detailed.py`

快速 8 并发功能验证：
```bash
cd /path/to/310p-vllm-ascend && \
source /home/nin/Workspace/.venv/bin/activate && \
python3 deploy/tests/perf_benchmark_detailed.py
```

---

**维护者**: CANNBot model-infer-optimize  
**日期**: 2026-07-08  
**关联报告**: `deploy/docs/reports/NIGHTLY_PERFORMANCE_COMPARISON_20260707.md`
