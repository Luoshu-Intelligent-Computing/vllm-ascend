# 性能基准测试报告（详细版）

**测试日期**: 2026-06-29  
**模型**: Qwen3.6-35B-A3B-w8a8  
**硬件**: Atlas 300I Duo (310P3 × 2，~87 GB HBM)  
**框架**: vllm-ascend 26.0.0  
**服务配置**:
- max_model_len: 131072 (128K)
- max_num_batched_tokens: **1024**（精度修复参数）
- tensor_parallel_size: 2
- gpu_memory_utilization: 0.75
- enable_chunked_prefill: true
- enable_thinking: False

**测试方法**: 流式 API，精确测量首 token 时间（TTFT），每次重复 3 次取中位数

---

## Part 1: Prefill + E2E 综合性能

| 标签 | 实际输入(tokens) | max_out | TTFT(ms) | E2E(ms) | 实际输出(tokens) | Prefill(t/s) | Decode(t/s) | ms/token |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| in256_out128 | 301 | 128 | **948** | 1935 | 32 | 317.6 | 31.4 | 31.8 |
| in1k_out128 | 1197 | 128 | **2839** | 3858 | 32 | 421.6 | 31.4 | 31.8 |
| in4k_out128 | 4781 | 128 | **8531** | 9529 | 31 | 560.4 | 31.2 | 32.1 |
| in8k_out128 | 9555 | 128 | **15094** | 16103 | 32 | 633.0 | 30.7 | 32.5 |
| in16k_out128 | 19117 | 128 | **31542** | 32479 | 28 | 606.1 | 30.0 | 33.3 |
| in32k_out128 | 38227 | 128 | **64520** | 65496 | 28 | 592.5 | 28.7 | 34.8 |
| in256_out512 | 301 | 512 | 904 | 1922 | 31 | 332.9 | 31.4 | 31.8 |
| in1k_out512 | 1197 | 512 | 2901 | 3919 | 32 | 412.6 | 31.4 | 31.8 |
| in4k_out512 | 4781 | 512 | 8212 | 9208 | 31 | 582.2 | 31.1 | 32.1 |

### 关键观察

**Prefill 吞吐（t/s）随输入长度的变化**：

| 输入规模 | Prefill 吞吐 | 说明 |
|---------|------------|------|
| 256 tokens | 318 t/s | 短 prompt，ChunkedPrefill 调度开销比例高 |
| 1K tokens | 422 t/s | 开始提升 |
| 4K tokens | 560 t/s | 继续提升 |
| 8K tokens | 633 t/s | 峰值附近 |
| 16K tokens | 606 t/s | 微降（chunk 数增多，overhead 累积）|
| 32K tokens | 593 t/s | 趋于稳定 |

**峰值 Prefill 吞吐：~633 t/s（8K 输入规模）**

> 注：相比 max_num_batched_tokens=2048 时的峰值 ~960 t/s（见 PERFORMANCE_BASELINE_20260625.md），
> 降低约 34%。这是为保证长上下文精度必须付出的代价（ATB 算子在 2048 时有精度问题）。

---

## Part 2: 纯 Decode 速度

**Prefill baseline（短 prompt，decode=1）：801 ms（中位数）**

| Decode 目标 | 实际输出 | Decode 时间(ms) | Decode(t/s) | ms/token |
|:---:|:---:|:---:|:---:|:---:|
| 32 | 32 | 973 | **32.9** | 30.4 |
| 64 | 64 | 1990 | **32.2** | 31.1 |
| 128 | 128 | 4075 | **31.4** | 31.8 |
| 256 | 256 | 8095 | **31.6** | 31.6 |
| 512 | 512 | 16217 | **31.6** | 31.7 |

**Decode 速度非常稳定：~31.5–32.9 t/s，约 31–32 ms/token**，与输出长度无关。

---

## 性能指标汇总

| 指标 | 数值 |
|------|------|
| **TTFT（256 token prompt）** | ~948 ms |
| **TTFT（1K token prompt）** | ~2839 ms |
| **TTFT（4K token prompt）** | ~8531 ms |
| **TTFT（8K token prompt）** | ~15094 ms |
| **TTFT（16K token prompt）** | ~31542 ms |
| **TTFT（32K token prompt）** | ~64520 ms |
| **峰值 Prefill 吞吐** | ~633 t/s（8K 规模） |
| **Decode 速度** | **~31.5 t/s（~31.7 ms/token）** |
| **E2E 延迟（256in+32out）** | ~1.9 s |
| **E2E 延迟（4Kin+128out）** | ~9.5 s |
| **E2E 延迟（32Kin+128out）** | ~65.5 s |
| 显存占用（128K 空闲）| ~31.7 GB/卡（73.8%）|

---

## 对比说明

| 参数 | max_batched=2048（旧）| max_batched=1024（当前）| 变化 |
|------|:---:|:---:|:---:|
| 峰值 Prefill | ~960 t/s | ~633 t/s | **-34%** |
| Decode 速度 | ~31.5 t/s | ~31.5 t/s | 无变化 |
| 长上下文精度 | ❌ 乱码 | ✅ 正常 | 修复 |

Decode 速度不受 `max_num_batched_tokens` 影响（decode 阶段每步只处理 1 token），Prefill 吞吐下降是因为每个 chunk 更小（1024 vs 2048），需要更多调度轮次。

---

## 原始数据

详细 JSON 已保存：`/tmp/perf_detailed.json`

---

**测试人**: CANNBot model-infer-optimize  
**更新时间**: 2026-06-29
