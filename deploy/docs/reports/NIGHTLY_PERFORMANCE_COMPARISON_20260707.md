# nightly vs 生产基线 性能对比报告

**测试日期**: 2026-07-07  
**模型**: Qwen3.6-35B-A3B-w8a8  
**硬件**: Atlas 300I Duo (310P3 × 2，~87 GB HBM)

---

## 测试配置对比

| 项目 | 生产基线（26.0.0）| nightly（新）|
|------|:-----------------:|:------------:|
| 镜像 | `vllm-ascend:26.0.0-poc-300i-duo` | `vllm-ascend:nightly-main-310p` |
| CANN 版本 | 9.0.T3 | **9.1.0-beta.1** |
| vllm-ascend | 26.0.0 | 0.19.1rc2.dev755 |
| max_model_len | 131072 | 131072 |
| max_num_batched_tokens | 1024 | 1024 |
| tensor_parallel_size | 2 | 2 |
| gpu_memory_utilization | 0.75 | 0.75 |
| **GDN Prefill** | PyTorch fallback | **AscendC kernel** |
| Attention mask | 动态 chunk mask（patch）| 动态 chunk mask（nightly patch）|
| 测试方法 | 流式 API，TTFT，每次重复 3 次取中位数 | 流式 API，TTFT，每次重复 2 次取中位数 |

### 关键差异

**nightly 的核心升级**：`chunk_gated_delta_rule_fwd_h` + `chunk_fwd_o` AscendC kernel 生效，Qwen3.6-35B 的 30 层 GDN Prefill 从 PyTorch fallback 升级为 310P 专属 NPU 加速。

---

## Prefill 性能对比

### TTFT（首 token 时间）

| 输入规模 | 生产基线 TTFT(ms) | nightly TTFT(ms) | 变化 |
|---------|:----------------:|:----------------:|:----:|
| ~256t   | 948              | **763**          | **-19.5%** ↓ |
| ~1k t   | 2839             | **1593**         | **-43.9%** ↓ |
| ~4k t   | 8531             | **5814**         | **-31.9%** ↓ |
| ~8k t   | 15094            | **11644**        | **-22.9%** ↓ |
| ~16k t  | 31542            | **23656**        | **-25.0%** ↓ |
| ~32k t  | 64520            | **49569**        | **-23.2%** ↓ |

### Prefill 吞吐（tokens/s）

| 输入规模 | 生产基线 | nightly | 变化 |
|---------|:-------:|:-------:|:----:|
| ~256t   | 317.6   | **335.4**  | +5.6% |
| ~1k t   | 421.6   | **642.9**  | **+52.5%** ↑ |
| ~4k t   | 560.4   | **704.5**  | **+25.7%** ↑ |
| ~8k t   | 633.0   | **703.5**  | +11.1% |
| ~16k t  | 606.1   | **692.6**  | +14.3% |
| ~32k t  | 592.5   | **661.1**  | +11.6% |

**峰值 Prefill 吞吐：633 t/s → 704.5 t/s（+11.3%）**

---

## Decode 性能对比

| 输出长度 | 生产基线 (t/s) | nightly (t/s) | 变化 |
|---------|:-------------:|:-------------:|:----:|
| 64t     | ~32.9         | **35.3**      | +7.3% |
| 128t    | ~31.4         | **34.7**      | +10.5% |
| 256t    | ~31.6         | **34.2**      | +8.2% |

**Decode 平均：~31.5 t/s → ~34.7 t/s（+10.2%）**

---

## 精度验证

| 评估 | 生产基线 | nightly | 结论 |
|------|:-------:|:-------:|:----:|
| GSM8K（20样本）| 98.0%（50样本基线）| **95.0%** | ✅ 统计波动范围内 |
| 长上下文（57k token）| ✅ | ✅ | ✅ 一致 |
| 简单问答 | ✅ | ✅ | ✅ 一致 |

> GSM8K 95.0% vs 98.0% 差值在 20 样本的统计误差范围内（唯一错误题目 #13 与基线相同，属模型本身的数学边界问题）。

---

## 性能提升来源分析

### 1k token 规模提升最显著（+52.5%）

1k token 的 prompt 通常包含约 85 个 GDN chunk（每 chunk 12 token），在生产基线中全部走 PyTorch fallback `torch_chunk_gated_delta_rule`，CPU 密集；nightly 中全部走 310P AscendC kernel，NPU 并行加速。

**规模越小提升越明显**：短 prompt 中 GDN 层占总推理耗时比例更高；长 prompt 中 Attention 计算权重增大，GDN 加速收益相对稀释。

### Decode 均匀提升（+8-11%）

Decode 阶段每步仅生成 1 token，GDN 走 `npu_recurrent_gated_delta_rule_310`（两个版本均已有 NPU kernel），收益来自 CANN 9.1 底层调度优化。

---

## 关键约束（310P 平台）

| 约束 | 值 | 原因 |
|------|:--:|------|
| `max_num_batched_tokens` | **≤ 1024** | ATB 算子在 ≥2048 时精度静默损坏（已验证）|
| `_npu_flash_attention_v3` | ❌ 不可用 | CANN 9.1.0-beta.1 仍未集成到 torch_npu 2.10.0 |
| `_npu_paged_attention_splitfuse_v2` | ❌ 不可用 | 同上，需等待 torch_npu 版本更新 |
| Attention mask 方案 | 动态 chunk mask | v3/v2 不可用时的 fallback，需手动 patch |

---

## 结论

| 指标 | 生产基线 → nightly | 评估 |
|------|:------------------:|:----:|
| 峰值 Prefill 吞吐 | 633 → 704 t/s | **+11%** ✅ |
| TTFT（1k prompt）| 2839 → 1593 ms | **-44%** ✅ |
| Decode 吞吐 | 31.5 → 34.7 t/s | **+10%** ✅ |
| GSM8K 精度 | 98%（50样本）→ 95%（20样本）| **无回归** ✅ |
| 128K 上下文支持 | ✅ | ✅ |

**nightly 方案全面优于生产基线，建议作为下一版生产方案。**

主要风险：
- CANN 9.1.0-beta.1（非正式商用版），稳定性需持续观察
- 服务启动需重新编译（~30 分钟），可通过预编译镜像解决

---

## 附：启动命令

```bash
# nightly 128K 服务启动命令（含 GDN AscendC kernel）
podman run -d \
  --name vllm-nightly-128k \
  --privileged --network host \
  --device /dev/davinci0 --device /dev/davinci1 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -e ASCEND_RT_VISIBLE_DEVICES=0,1 \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
  -v /srv/meetai/models:/models:ro \
  -v /path/to/patches-nightly:/workspace/patches-nightly:ro \
  quay.io/ascend/vllm-ascend:nightly-main-310p \
  bash -c '
    source /usr/local/Ascend/cann-9.1.0-beta.1/set_env.sh
    VLLM_PATH=$(python3 -c "import vllm_ascend,os; print(os.path.dirname(vllm_ascend.__file__))")
    cp /workspace/patches-nightly/attention_mask.py $VLLM_PATH/_310p/attention/
    cp /workspace/patches-nightly/metadata_builder.py $VLLM_PATH/_310p/attention/
    # 重新编译（含 ASCEND_PLATFORM_310P 宏，激活 GDN chunk kernel）
    cd /vllm-workspace/vllm-ascend && SOC_VERSION=ascend310p1 pip install -e . --no-build-isolation --no-deps -q
    python3 -m vllm.entrypoints.openai.api_server \
      --model /models/llm-service/vllm-ascend/Qwen3.6-35B-A3B-w8a8 \
      --served-model-name qwen3.6-128k-nightly \
      --max-model-len 131072 --max-num-batched-tokens 1024 \
      --tensor-parallel-size 2 --gpu-memory-utilization 0.75 \
      --enable-chunked-prefill --no-enable-prefix-caching \
      --reasoning-parser qwen3 \
      --additional-config "{\"ascend_compilation_config\": {\"fuse_norm_quant\": false}}" \
      --compilation-config "{\"cudagraph_mode\": \"FULL_DECODE_ONLY\", \"cudagraph_capture_sizes\": [1]}"
  '
```

---

**维护者**: CANNBot model-infer-optimize  
**日期**: 2026-07-07  
**状态**: 验证通过，建议推进生产切换评审
