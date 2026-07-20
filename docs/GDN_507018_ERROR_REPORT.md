# GDN 507018 错误报告

**时间**: 2026-07-17  
**环境**: OrangePi（Atlas 200I Pro，310P1 SoC，RC 模式）  
**镜像**: `registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend:310p-opt-openeuler-20260709`

---

## 错误现象

服务成功启动（OOM 修复后），第一次推理请求即触发崩溃，错误码 507018。

**服务端日志**：
```
(EngineCore pid=287) ERROR 07-17 03:37:04 [core.py:1197] RuntimeError: ACL stream synchronize failed, error code:507018
(APIServer pid=248) ERROR 07-17 03:37:04 [async_llm.py:704] vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue.
```

---

## 完整调用栈

```
acl_graph.py:823 → __call__
  → acl_graph.py:894 → fx_run_eagerly    # npugraph 内落回 eager 模式
    → acl_graph.py → fx_forward
      → qwen_gdn_linear_attn.py:1731 → qwen_gdn_attention_core
        → gdn_310.py:475 → _forward_core
          → chunk_gated_delta_rule.py:527 → chunk_gated_delta_rule_310
            → q, k, v, g, beta, cu_seqlens.to(torch.int64).cpu(), CHUNK_SIZE
                                             ^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: ACL stream synchronize failed, error code:507018
```

---

## 根因分析

### 触发位置
**文件**：`/vllm-workspace/vllm-ascend/vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py:527`

**关键代码**：
```python
# line 527
q, k, v, g, beta, cu_seqlens.to(torch.int64).cpu(), CHUNK_SIZE
```

`cu_seqlens.to(torch.int64).cpu()` 在 ACL 图执行上下文中调用了 `.cpu()` ，将 tensor 移到 CPU。

### 为什么失败

启动参数包含 `"enable_npugraph_ex": true`，激活了 ACL 图模式（npugraph_ex）。在 ACL 图上下文中调用 `.cpu()` 会触发 AICPU 同步操作，在 310P1 SoC RC 模式 + CANN 9.1 下该操作失败，返回 507018（AICPU stream synchronize failed）。

### EP 模式为何不失败

- EP 模式（ails-a1，300I Duo，310P3）使用相同配置时正常运行
- 推测原因：EP 模式的 AICPU 上下文与 RC 模式不同，`.cpu()` 操作在 PCIe 设备上的处理路径有差异

---

## 调用链关系

```
GDN 层（30 层线性循环注意力）
  └── qwen_gdn_linear_attn.py → qwen_gdn_attention_core
        └── gdn_310.py → _forward_core（310P 专用实现）
              └── chunk_gated_delta_rule.py → chunk_gated_delta_rule_310
                    └── 调用 AscendC kernel（vllm_ascend_C.so）
                          → cu_seqlens.to(torch.int64).cpu()  ← 问题点
```

---

## ATB 日志证据

`/root/ascend/log/atb/atb_287_20260717032452.log` 显示：
- Warmup 阶段（03:34:23）`PagedAttentionDecoderNzMaskKernel` 正常执行
- 507018 出现在首次真实推理（03:37:04）
- 说明 warmup 不触发 GDN 路径，实际推理才触发

---

## 验证方向

### 方案 A：禁用 npugraph_ex
```bash
--additional-config '{"ascend_compilation_config": {"fuse_norm_quant": false, "enable_npugraph_ex": false}}'
```
预期：`.cpu()` 在非图模式下可能正常工作（待验证）

### 方案 B：修复 chunk_gated_delta_rule.py
在 GDN 算子里避免在 ACL 图上下文调用 `.cpu()`：
```python
# 原始（有问题）
cu_seqlens.to(torch.int64).cpu()

# 修复方向：提前在图外 pin 到 CPU，或用 npu tensor 替换
cu_seqlens_cpu = cu_seqlens.to(torch.int64).cpu()  # 在图捕获前执行
```

### 方案 C：对 GDN 层单独禁用图模式
通过条件判断，在 GDN 前向传播时退出图捕获上下文。

---

## 相关文件路径

| 文件 | 路径 |
|------|------|
| GDN 前向 | `/vllm-workspace/vllm-ascend/vllm_ascend/_310p/ops/fla/gdn_310.py:475` |
| 问题代码 | `/vllm-workspace/vllm-ascend/vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py:527` |
| GDN 注意力 | `/vllm-workspace/vllm/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1731` |
| ACL 图 | `/usr/local/python3.12.13/.../torch_npu/dynamo/npugraph_ex/_acl_concrete_graph/acl_graph.py` |

---

## 启动参数（复现配置）

```bash
vllm serve /models/Qwen3.6-35B-A3B-w8a8 \
  --served-model-name qwen3.6 \
  --host 0.0.0.0 --port 38081 -tp 1 \
  --max-model-len 131072 --max-num-seqs 1 --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.75 --dtype float16 --kv-cache-dtype auto \
  --trust-remote-code --enable-chunked-prefill --no-enable-prefix-caching \
  --reasoning-parser qwen3 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1]}' \
  --additional-config '{"ascend_compilation_config": {"fuse_norm_quant": false, "enable_npugraph_ex": true}}'
  # ↑ enable_npugraph_ex=true 是触发因素
```

---

**记录人**: CANNBot  
**日期**: 2026-07-17
