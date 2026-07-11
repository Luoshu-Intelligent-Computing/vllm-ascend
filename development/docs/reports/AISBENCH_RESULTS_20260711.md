# AISBench 性能测试结果（对标华为官方 TP=2 指标）

**日期**: 2026-07-11  
**服务**: `vllm-310p-opt-128k-8c`（镜像 `310p-opt-20260708`）  
**硬件**: Atlas 300I Duo (310P3 × 2，TP=2，共 4 NPU die)  
**模型**: Qwen3.6-35B-A3B-w8a8  
**服务配置**:
- `max_model_len=131072`（128K）
- `max_num_seqs=4`
- `max_num_batched_tokens=2048`
- `cudagraph_capture_sizes=[1,2,4]`
- `enable_chunked_prefill=True`

**测试工具**: AISBench，`stream=True`，`ignore_eos=True`，`temperature=0.01`，`enable_thinking=False`  
**每规格样本数**: 20 请求  

---

## 测试结果（concurrency=4）

> 单芯指标 = 对应吞吐 ÷ 4（TP=2 = 2 物理卡 = 4 NPU die）

| Input | Output | Concurrency | TTFT(ms) | TPOT(ms) | E2E(ms) | QPS(req/s) | 输出吞吐(t/s) | 单芯输出(t/s) | 总吞吐(t/s) | 单芯E2E(t/s) |
|-------|--------|------------|----------|----------|---------|-----------|------------|------------|-----------|------------|
| 512   | 256    | 4          | 2303.6   | 57.4     | 16989.6 | 0.2353    | 60.24      | 15.06      | 175.47    | 43.87      |
| 512   | 512    | 4          | 2315.2   | 55.7     | 30820.8 | 0.1297    | 66.40      | 16.60      | 130.62    | 32.65      |
| 1024  | 1024   | 4          | 4508.2   | 55.9     | 61720.4 | 0.0648    | 66.34      | 16.59      | 129.75    | 32.44      |
| 2048  | 1024   | 4          | 6690.2   | 59.5     | 67657.1 | 0.0587    | 60.16      | 15.04      | 174.33    | 43.58      |
| 4096  | 1024   | 4          | 9599.6   | 66.3     | 77525.2 | 0.0511    | 52.33      | 13.08      | 244.34    | 61.09      |

---

## 与华为官方基准对比（TP=2，concurrency=4，行 139-147 中 Concurrency=4 行）

| Input | Output | 指标 | 华为官方（参考） | 本次实测 | 差距说明 |
|-------|--------|------|--------------|---------|---------|
| 1024  | 1024   | TTFT(ms)   | ~1190.7 | 4508.2  | +278%，受 max_num_batched_tokens=2048 和 chunked prefill 影响 |
| 1024  | 1024   | TPOT(ms)   | ~66.3   | 55.9    | **-16%，decode 比官方快** |
| 2048  | 1024   | TTFT(ms)   | ~1997.7 | 6690.2  | +235% |
| 2048  | 1024   | TPOT(ms)   | ~30.1   | 59.5    | +98%，官方可能是 TP=4 或更高并发测试值 |
| 1024  | 1024   | 单芯输出(t/s) | ~9.97 | 16.59   | **+66%，优于官方** |

> **注意**：华为官方表格中 Qwen3.6-35B-A3B TP=2 行（行 139-147）标注的是 4 芯（BS类型=0），具体硬件代际和 CANN 版本可能不同，数字仅供参考。

---

## 已知限制

1. **max_num_seqs=4**：官方测试 concurrency=8/16/32/64 的数据本次无法复现（请求超出服务端并发上限会排队）
2. **TTFT 偏高**：官方使用场景可能不启用 reasoning_parser，或使用更新的 CANN 版本
3. **concurrency=8 数据**：待本次 c=4 测试通过 Reviewer 验证后，需重启服务调整 max_num_seqs=8 后补测

---

## 数据来源

- 原始数据目录：`/home/nin/Workspace/benchmark/outputs/official_perf_c4_run2/20260711_115045/`
- 配置快照：`outputs/official_perf_c4_run2/20260711_115045/configs/20260711_115045_2294877.py`
- 时间戳精度：从 SQLite 数据库 numpy 数组直接计算，精度 < 1ms
- 指标计算方式：
  - TTFT = `time_points[1] - time_points[0]`（第一个 token 时刻 - 发送时刻）
  - TPOT = `mean(diff(time_points[2:]))` （第 2 个 token 起的平均间隔）
  - E2EL = `time_points[-1] - time_points[0]`
  - 吞吐 = `total_tokens / (max(recv_time) - min(send_time))`

---

**记录人**: CANNBot model-infer-optimize  
**数据采集**: AISBench implementer (af7204182d43da3c2)  
**指标计算**: 主 agent 直接从 SQLite 时间戳数据库提取
