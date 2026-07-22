# 技术简报：GDN Kernel 507018 错误调查报告

**日期**: 2026-07-20
**硬件**: Atlas 200I Pro（310P1 SoC，RC 模式，87.5 GB HBM 共享内存）
**模型**: Qwen3.6-35B-A3B-w8a8（混合 GDN + 标准 Attention）
**测试镜像**:
- `vllm-ascend:dev-26.0.0.poc.20260413-9.0.T3.B030`（老 POC，CANN 9.0）✅ 可用
- `llm-service-vllm-ascend:310p-opt-openeuler-20260709`（新镜像，CANN 9.1）❌
- `llm-service-vllm-ascend:310p-opt-20260713`（最新镜像，CANN 9.1）❌

---

## 一、问题概述

新镜像（CANN 9.1）服务启动成功，但首次推理请求必然崩溃，错误码 507018。
老 POC 镜像（CANN 9.0）同模型、同硬件正常运行，decode 吞吐 ~20.3 t/s。

---

## 二、错误码说明

| 错误码 | 含义 |
|-------|------|
| `507018` | `ACL stream synchronize failed` / `rtsLaunchKernelWithHostArgs failed` |
| `0x2a` | AICPU 异常，函数 `copy_between_host_and_device_opapi` |

两个错误码贯穿所有崩溃场景，指向同一根因：**在 ACL stream 上下文内发生了不被允许的 host-device 数据传输**。

---

## 三、完整测试记录

### 测试 1：原始配置（20260709 镜像，npugraph_ex=true）

**配置**：
```bash
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1]}'
--additional-config '{"ascend_compilation_config": {"fuse_norm_quant": false, "enable_npugraph_ex": true}}'
```

**错误栈**：
```
(EngineCore pid=287) ERROR [core.py:1197]
  File "acl_graph.py", line 823, in __call__
    return self.fx_run_eagerly(*args, **kwargs)
  File "acl_graph.py", line 894, in fx_run_eagerly
    return self.fx_forward(*args, **kwargs)
  File "qwen_gdn_linear_attn.py", line 1731, in qwen_gdn_attention_core
  File "gdn_310.py", line 475, in _forward_core
  File "chunk_gated_delta_rule.py", line 527, in chunk_gated_delta_rule_310
    q, k, v, g, beta, cu_seqlens.to(torch.int64).cpu(), CHUNK_SIZE
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: ACL stream synchronize failed, error code:507018

DEVICE[0] PID[287]:
  Message info[0]: RTS_HWTS: Aicpu exception, slot_id=4, stream_id=11
  Other info[0]: error code=0x2a (function copy_between_host_and_device_opapi)
```

**根因**：`chunk_gated_delta_rule_310` 函数在 npugraph_ex 的 `fx_run_eagerly` 上下文中调用了 `cu_seqlens.to(torch.int64).cpu()`，触发 NPU→CPU 同步拷贝。CANN 9.1 在 ACL stream SCHEDULE 阶段不允许此操作。

---

### 测试 2：修复 `.cpu()` 调用（gdn_310.py + gdn_attn_builder_310.py，npugraph_ex=true）

**修复内容**：
- `gdn_attn_builder_310.py:build()` 在图执行前预计算 `non_spec_query_start_loc_cpu`（CPU tensor）
- `gdn_310.py:484` 改为 `cu_seqlens=non_spec_query_start_loc_cpu`

**错误栈**：
```
(EngineCore pid=286) ERROR [core.py:1197]
  File "gdn_310.py", line 472, in _forward_core
RuntimeError: The Inner error is reported as above.
The current working operator name is aclnnMatmul.

[rank0]: NPU function error: call aclnnMatmul failed, error code is 507018
         rtsLaunchKernelWithHostArgs failed, runtime result = 507018
         Other info: error code=0x2a (function copy_between_host_and_device_opapi)
```

**根因变化**：`.cpu()` 问题已解决（调用点消失），但 `chunk_gated_delta_rule_310` 内部的 AscendC 自定义 matmul 算子调用 `rtsLaunchKernelWithHostArgs` 失败。

---

### 测试 3：禁用 npugraph_ex（enable_npugraph_ex=false）

**配置**：同测试 2 修复 + `enable_npugraph_ex: false`

**错误栈**：
```
(EngineCore pid=285) ERROR [core.py:1197]
  File "proxy_tensor.py", line 1507, in wrapped
RuntimeError: The Inner error is reported as above.
The current working operator name is aclnnMatmul.
error code is 507018
```

**结论**：禁用 npugraph_ex 后仍然崩溃，错误位置从 `npu_fx_compiler.py` 变为 `proxy_tensor.py`（torch dynamo 路径），但根因相同：`rtsLaunchKernelWithHostArgs` 在 RC 模式下失败。

---

### 测试 4：完全禁用编译（--enforce-eager）

**配置**：`--enforce-eager` + `enable_npugraph_ex: false`

**错误栈**：
```
(EngineCore pid=286) ERROR [core.py:1197]
  File "qwen_gdn_linear_attn.py", line 1731, in qwen_gdn_attention_core
  File "gdn_310.py", line 472, in _forward_core
RuntimeError: ACL stream synchronize failed, error code:507018

rtsLaunchKernelWithHostArgs failed, runtime result = 507018
error code=0x2a (function copy_between_host_and_device_opapi)
```

**结论**：即使完全 eager 模式（无任何图优化），`chunk_gated_delta_rule_310` 内的 AscendC kernel 启动仍然失败。问题与图模式无关，是 CANN 9.1 kernel 执行层的兼容性问题。

---

### 测试 5：最新镜像（310p-opt-20260713，enforce-eager，无修复）

**错误栈**：
```
(EngineCore pid=285) ERROR [core.py:1197]
  File "qwen_gdn_linear_attn.py", line 1731, in qwen_gdn_attention_core
  File "gdn_310.py", line 359, in _forward_core
RuntimeError: ACL stream synchronize failed, error code:507018

synchronize stream failed, runtime result = 507018
error code=0x2a (function copy_between_host_and_device_opapi)
```

**结论**：`310p-opt-20260713` 镜像同样未修复此问题，原始 `.cpu()` 调用仍然存在，且 AscendC kernel 层的兼容性问题依然存在。

---

## 四、根因分析

### 4.1 调用链

```
vllm serve (prefill 请求)
  └── qwen_gdn_linear_attn.py:1731 → qwen_gdn_attention_core
        └── gdn_310.py:_forward_core
              └── chunk_gated_delta_rule_310(cu_seqlens=NPU tensor)
                    ├── [Layer 1] chunk_gated_delta_rule.py:527
                    │     cu_seqlens.to(torch.int64).cpu()   ← 触发 D2H，第一道障碍
                    └── [Layer 2] 内部 AscendC matmul kernel
                          rtsLaunchKernelWithHostArgs()       ← 第二道障碍
```

### 4.2 双重障碍

| 障碍 | 位置 | 错误信息 | 修复状态 |
|------|------|---------|---------|
| 障碍1 | `chunk_gated_delta_rule.py:527` `.cpu()` | `copy_between_host_and_device_opapi` | ✅ Python 层可修复 |
| 障碍2 | `chunk_gated_delta_rule_310` 内部 AscendC kernel | `rtsLaunchKernelWithHostArgs failed` | ❌ 需要 CANN 层修复 |

### 4.3 技术根因

`rtsLaunchKernelWithHostArgs` 是 AscendC kernel 的 Host 参数传递接口。在 310P1 SoC RC 模式下（CPU 与 NPU 共享物理内存，但有独立虚地址空间），CANN 9.1 对此接口加入了更严格的 stream fence 要求，导致原本在 CANN 9.0 和 EP 模式（PCIe 外接）下正常工作的调用失败。

**对比**：

| 环境 | CANN 版本 | 模式 | 结果 |
|------|----------|------|------|
| Atlas 200I Pro（香橙派） | 9.0.T3 | RC | ✅ 正常 |
| Atlas 200I Pro（香橙派） | 9.1.0-beta | RC | ❌ 507018 |
| ails-a1（300I Duo） | 9.1.0 | EP | ✅ 正常 |

---

## 五、已实施的修复

### 修复1：RC 模式检测（utils.py）

**文件**：`vllm_ascend/utils.py`
**问题**：新镜像无 `lspci`，`is_rc_device()` 返回 False，导致 KV cache 计算路径错误
**修复**：添加 `/sys/bus/pci/devices` fallback + 默认 True for 310P
**状态**：✅ 已验证有效

### 修复2：KV Cache OOM（worker_310p.py）

**文件**：`vllm_ascend/_310p/worker_310p.py`
**问题**：CANN 9.1 启动时预占 ~37GB RAM，psutil.available 从 ~83GB 降至 ~50GB，导致 KV cache 计算为负
**修复**：`determine_available_memory()` 改用 psutil delta（init 前后差值）计算模型 overhead
**状态**：✅ 已验证，KV cache = 11.26 GiB 正常

### 修复3（部分）：GDN cu_seqlens CPU 转换（gdn_310.py + gdn_attn_builder_310.py）

**文件**：`vllm_ascend/_310p/ops/fla/gdn_310.py`, `vllm_ascend/_310p/ops/gdn_attn_builder_310.py`
**问题**：`chunk_gated_delta_rule_310` 传入 NPU tensor，内部 `.cpu()` 在 ACL stream 上下文触发 507018
**修复**：`gdn_attn_builder_310.py:build()` 预计算 `non_spec_query_start_loc_cpu`，`gdn_310.py:484` 使用 CPU 副本
**状态**：⚠️ 解决了障碍1，障碍2（`rtsLaunchKernelWithHostArgs`）需要 CANN 层修复

---

## 六、阻塞点

**问题**：`chunk_gated_delta_rule_310` 内部的 AscendC 自定义算子在 CANN 9.1 + 310P1 RC 模式下，通过 `rtsLaunchKernelWithHostArgs` 启动时失败，错误码 507018。

**影响范围**：所有包含 GDN 层的模型（Qwen3.6 的线性循环注意力层），在 310P1 RC 模式 + CANN 9.1 下不可用。

**请华为确认**：

1. CANN 9.1 在 310P1 RC 模式下，`rtsLaunchKernelWithHostArgs` 是否有新增约束？
2. `chunk_gated_delta_rule_310` 的 AscendC kernel 是否需要针对 RC 模式重新编译或调整调用方式？
3. 是否有已知的 CANN 9.1 配置参数可以使 `rtsLaunchKernelWithHostArgs` 在 RC 模式下正常工作？
4. CANN 9.0（`9.0.T3.B030`）对此接口的处理是否与 9.1 存在差异？

---

## 七、当前部署状态

**临时方案**：继续使用老 POC 镜像（CANN 9.0）+ 128K 上下文 patch

| 指标 | 值 |
|------|-----|
| 镜像 | `vllm-ascend:dev-26.0.0.poc.20260413-9.0.T3.B030` |
| 最大上下文 | 131072 tokens |
| KV cache | 可用（RC OOM 修复已回移植） |
| Decode 吞吐 | ~20.3 t/s（npugraph_ex + FULL_DECODE_ONLY） |
| Prefill 吞吐 | ~243–315 t/s |

**阻塞升级到新镜像的问题**：GDN kernel `rtsLaunchKernelWithHostArgs` 在 CANN 9.1 RC 模式失败（507018），需华为修复。

---

## 八、环境信息

```
芯片型号:   Atlas 200I Pro（310P1 SoC，RC 模式）
单 die HBM: ~87.5 GB（CPU + NPU 共享）
CANN 版本:  9.1.0-beta.1（新镜像，存在问题）
           9.0.T3.B030（老镜像，正常）
PyTorch:    2.10.0 + torch_npu
vLLM:       0.23.0
模型:       Qwen3.6-35B-A3B-w8a8（W8A8 量化，38 GB）
部署配置:   TP=1，单卡，max_model_len=131072
```

---

*记录人: 联合调试 2026-07-20*
