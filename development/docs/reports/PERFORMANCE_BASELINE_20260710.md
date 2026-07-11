# 性能基线报告（三轮）

**模型**: Qwen3.6-35B-A3B-w8a8  
**硬件**: Atlas 300I Duo (310P3 × 2，共 ~87 GB HBM)  
**镜像**: `310p-opt-20260708`（含 GDN AscendC kernel，源码烘焙）  
**测试时间**: 2026-07-10  
**框架**: vllm-ascend（fork，feat/310p-opt）

---

## 最终生产配置

```bash
--max-model-len 131072          # 128K 上下文
--max-num-batched-tokens 2048   # ATB 精度安全值
--max-num-seqs 4                # 与 cudagraph capture_sizes 对齐
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 4]}'
--enable-chunked-prefill
-tp 2
```

---

## 一、Prefill 性能

**测试方法**: benchmark-live-server.py，mode=prefill，concurrency=1，3轮取均值  
**说明**: 表中 `prefill t/s` 为 `prompt_tokens / e2e_wall_time`（含短 decode 分母，略低于纯 prefill 速度）

| prompt 规模 | prompt tokens | E2E avg (ms) | Prefill 吞吐(t/s) |
|------------|--------------|-------------|-----------------|
| ~240t      | 240          | 766         | 315             |
| ~942t      | 942          | 998         | 943             |
| **~1554t** | 1,554        | 1,242       | **1,251（峰值）** |
| ~3067t     | 3,067        | 2,977       | 1,030           |
| ~6163t     | 6,163        | 7,062       | 873             |
| ~12139t    | 12,139       | 13,994      | 867             |
| ~24164t    | 24,164       | 30,146      | 802             |

**关键规律**:
- 峰值约 **1,251 t/s**，出现在 ~1.5k token 规模（单个 chunk 填满 2048 batched tokens）
- 长上下文（24k）约 **802 t/s**，仍为多 chunk 串行调度，无 OOM
- 相比旧基线（633 t/s，max_batched=1024，无 GDN kernel）提升约 **+99%**，来源：GDN AscendC kernel（主）+ max_batched 翻倍（次）

---

## 二、Decode 性能（单请求）

**测试方法**: benchmark-live-server.py，mode=decode，concurrency=1，纯 decode 速度通过扣除 prefill 时间估算

| prompt 规模 | 纯 Decode 吞吐 | 说明 |
|------------|-------------|------|
| ~240t      | ~29.4 t/s   | |
| ~942t      | ~30.8 t/s   | |
| ~3067t     | ~31.4 t/s   | |
| **专项测试**（20t prompt）| **~31.5 t/s（31.6 ms/token）** | 差值法精确测量 |

**结论**: Decode 单请求吞吐约 **31.5 t/s（31.6 ms/token）**，与 prompt 长度无关（KV cache 加速后不随上下文增长劣化）。

---

## 三、并发吞吐

### 3.1 cudagraph 优化前（capture_sizes=[1]，max_num_seqs=8）

| 并发数 | E2E 聚合(t/s) | Decode 聚合(t/s) | 单请求延迟(s) | 说明 |
|-------|-------------|----------------|------------|------|
| c=1   | 72.3        | 28.1           | 9.10       | 正常 |
| c=2   | 20.7        | 8.1            | 63.53      | ❌ eager fallback |
| c=4   | 37.6        | 14.6           | 69.88      | ❌ eager fallback |
| c=8   | 64.3        | 25.0           | 81.72      | ❌ eager fallback |

**根因**: `capture_sizes=[1]` 仅捕获 batch_size=1 的图，并发≥2 时 decode 退回 eager 模式，延迟膨胀 7-8x。

### 3.2 cudagraph 优化后（capture_sizes=[1,4]，max_num_seqs=4）

**3轮稳定性数据**（抖动 <5%）:

| 并发数 | Round 0 | Round 1 | Round 2 | 均值 | vs 优化前 | 单请求延迟(s) |
|-------|---------|---------|---------|------|---------|------------|
| c=1   | 70.3    | 72.9    | 72.9    | **72.0 t/s** | ≈持平 | 9.14 |
| c=2   | 91.2    | 91.9    | 92.2    | **91.8 t/s** | **+343%** | 14.32 |
| c=4   | 162.3   | 168.5   | 161.0   | **163.9 t/s** | **+336%** | 16.03 |

**说明**:
- c=2/4 均命中 batch_size=4 的 cudagraph，decode 走图模式
- c=4 为当前最优工作点：聚合 decode 63.8 t/s，单请求延迟仅 16s（vs c=1 的 9.1s，代价合理）
- c=8 受限于 `max_num_seqs=4`，不再测试

### 3.3 遗留问题

`capture_sizes=[1,4,8]` 在 TP=2 下触发 HCCL ACL capture event 限制，目前无法捕获 batch_size=8 的图。待上游修复后可进一步扩展到 c=8。

---

## 四、精度验证

| 评估方式 | 结果 | 配置 |
|---------|------|------|
| 多长度精度（8个长度点，含 chunk 边界）| ✅ 8/8 | batched=2048 |
| GSM8K 50样本 | ✅ 100%（50/50） | batched=2048，seqs=8 |

**关键精度修复（历史）**: `max_num_batched_tokens=1024` 是针对旧镜像的约束（旧镜像 >7k tokens 推理乱码）；新镜像 `310p-opt-20260708` 验证 2048 精度正常，上限待继续测试。

---

## 五、与旧基线对比

| 指标 | 旧基线（2026-06-29，旧镜像） | 当前基线（2026-07-10，新镜像） | 变化 |
|------|--------------------------|-----------------------------|------|
| 镜像 | 旧 nightly（无 GDN kernel） | `310p-opt-20260708`（含 GDN kernel） | - |
| max_batched_tokens | 1024 | 2048 | +100% |
| Prefill 峰值 | ~633 t/s | **~1,251 t/s** | **+98%** |
| Prefill @6k | ~593 t/s | ~873 t/s | **+47%** |
| Decode 单请求 | ~31.5 t/s | ~31.5 t/s | ≈持平 |
| 并发 c=2 E2E | 20.7 t/s | **91.8 t/s** | **+343%** |
| 并发 c=4 E2E | 37.6 t/s | **163.9 t/s** | **+336%** |
| max_model_len | 65536 | **131072** | +100% |

---

## 六、下一步优化方向

| 方向 | 预期收益 | 优先级 | 状态 |
|------|---------|--------|------|
| max_num_batched_tokens 4096 | Prefill 进一步提升 | P1 | 精度验证中 |
| HCCL cudagraph 上游修复（c=8） | 并发 c=8 图模式 | P2 | 待上游 |
| msprof profiling | 定量分析 GDN kernel 占比 | P2 | 待执行 |
| GDN chunk_size 优化 | Prefill 中等提升 | P3 | 待分析 |

---

**记录人**: CANNBot model-infer-optimize  
**数据来源**: `/tmp/baseline-prefill.json`, `/tmp/baseline-decode.json`, `/tmp/baseline-concurrent.json`, `/tmp/cudagraph-seqs4.json`, `/tmp/decode_benchmark.json`（Jul 9 旧数据，仅参考）
