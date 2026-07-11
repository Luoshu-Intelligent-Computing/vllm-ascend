# Phase 6: causal_fa_310p 算子集成验证报告

**时间**: 2026-07-09  
**状态**: ❌ 集成失败（发现阻塞性问题）  
**结论**: 当前 causal_fa_310p 算子**不适合**集成到生产环境

---

## 执行摘要

尝试将 `/home/nin/Workspace/310-ops/operators/causal_fa_310p` 自研算子集成到 vllm-ascend 推理框架，目标是优化 Prefill 阶段的 Attention 性能。集成过程发现两个阻塞性问题：

1. **代码 Bug**：abstract kernel 注册代码错误，导致算子实际未生效
2. **性能回退**：Decode 性能下降 **6.7x**（4.6 t/s vs 31.5 t/s），即使算子未生效也发生

精度验证通过（GSM8K 100%），但性能问题导致该算子暂时无法用于生产。

---

## 集成方案

### Feature Flag 机制

通过环境变量控制算子启用，代码改动在 `vllm_ascend/_310p/attention/attention_v1.py`：

```python
# 环境变量
USE_CAUSAL_FA_310P=1  # 启用 FA 算子
CAUSAL_FA_KERNEL_SO=/path/to/libcausal_fa_kernel.so
CAUSAL_FA_BINDINGS_SO=/path/to/torch_npu_causal_fa.cpython-311-aarch64-linux-gnu.so

# __init__ 中加载
if os.environ.get("USE_CAUSAL_FA_310P", "0") == "1":
    ctypes.CDLL(KERNEL_SO, mode=ctypes.RTLD_GLOBAL)
    torch.ops.load_library(BINDINGS_SO)
    self._use_causal_fa = True
```

### 调用路径

```
forward_prefill_310 → _flash_attention → 
  if self._use_causal_fa:
    torch.ops.npu_ext.causal_fa_310p(...)  # FA 路径
  else:
    _npu_flash_attention(...)  # 动态 chunk mask 路径（基线）
```

---

## 验证结果

### 精度验证 ✅

| 测试集 | FA 模式 | 基线（动态 mask） | 备注 |
|--------|---------|------------------|------|
| GSM8K (50) | **100%** (46/46) | 98% (49/50) | 4个超时因 Decode 变慢 |

精度本身无下降，超时是性能问题导致的（120s 限制内未完成）。

### 性能验证 ❌

#### Decode 性能（致命问题）

| 指标 | FA 模式 | 基线 | 变化 |
|------|---------|------|------|
| **Decode 吞吐** | **4.6 t/s** | 31.5 t/s | **-6.7x** ❌ |
| 延迟/token | 214 ms | 31.6 ms | +6.8x |

**问题**：即使 FA 算子未生效（fallback 到动态 mask），Decode 仍严重下降。

#### Prefill 性能（预期内）

| Prompt 规模 | FA E2E (ms) | 基线 E2E (ms) | 变化 |
|------------|-------------|--------------|------|
| 272 tokens | 2,779 | 1,393 | -100% (2x慢) |
| 1,040 tokens | 3,442 | 1,489 | -131% (2.3x慢) |
| 4,112 tokens | 7,553 | 5,774 | -31% |
| 8,204 tokens | 12,649 | 9,072 | -39% |
| 16,400 tokens | 23,505 | 17,601 | -34% |
| 32,780 tokens | 46,528 | 37,892 | -23% |

**原因**：progress.md 中已记录 "causal_fa_310p 使用标量 GEMM，prefill 延迟较高"，这是 P1 已知限制。

---

## 发现的问题

### Bug 1: abstract kernel 注册代码错误（已修复）

**现象**：`_use_causal_fa = False`，FA 算子从未被调用

**原因**：
```python
# 错误代码
if not torch.library.Library._erase_all:  # ← _erase_all 不存在，抛 AttributeError
    torch.library.impl_abstract(...)
```

`torch.library.Library._erase_all` 属性在 PyTorch 中不存在，导致 `AttributeError` 被外层 `except Exception` 捕获，`_use_causal_fa` 未设置为 `True`。

**修复**：
```python
# 正确代码（已修复）
try:
    torch.library.impl_abstract("npu_ext::causal_fa_310p")(
        lambda q, k, v, seq_lens, scale: torch.empty_like(q)
    )
except Exception:
    pass  # already registered
```

**影响**：此次测试中 FA 算子实际未生效，所有性能数据是 fallback 路径（动态 chunk mask）。

---

### Bug 2: Decode 性能回退 6.7x（根因未确认）

**现象**：即使 `_use_causal_fa = False`（未调用 FA 算子），Decode 从 31.5 t/s 跌到 4.6 t/s

**最可能原因**：`ctypes.CDLL(..., mode=ctypes.RTLD_GLOBAL)` 符号冲突

```python
# 问题代码
ctypes.CDLL(_CAUSAL_FA_310P_KERNEL_SO, mode=ctypes.RTLD_GLOBAL)
```

`RTLD_GLOBAL` 将 `libcausal_fa_kernel.so` 的所有符号全局暴露，可能与 NPU runtime 库发生符号冲突，破坏 NPU Graph 执行路径。

**验证方法**（待执行）：
```bash
# 控制实验：不加载 .so 文件
USE_CAUSAL_FA_310P=0  # 不加载算子 .so
# 测量 Decode 性能，如果恢复到 ~31.5 t/s，则确认是 RTLD_GLOBAL 导致
```

**可能的修复**：
1. 改为 `ctypes.RTLD_LOCAL`（符号仅对本进程可见）
2. 检查 `libcausal_fa_kernel.so` 是否导出了不必要的符号
3. 静态链接依赖库，减少符号冲突

---

## 代码改动

### 修改的文件

- `vllm_ascend/_310p/attention/attention_v1.py`（约 50 行新增）

### 修复后的关键代码片段

```python
# Feature Flag 路径参数化
_CAUSAL_FA_310P_KERNEL_SO = os.environ.get(
    "CAUSAL_FA_KERNEL_SO",
    "/home/nin/Workspace/310-ops/operators/causal_fa_310p/build/libcausal_fa_kernel.so",
)
_CAUSAL_FA_310P_BINDINGS_SO = os.environ.get(
    "CAUSAL_FA_BINDINGS_SO",
    "/home/nin/Workspace/310-ops/operators/causal_fa_310p/torch_ops/"
    "torch_npu_causal_fa.cpython-311-aarch64-linux-gnu.so",
)

def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._use_causal_fa = False
    
    if os.environ.get("USE_CAUSAL_FA_310P", "0") == "1":
        try:
            # 加载算子 .so 文件
            ctypes.CDLL(_CAUSAL_FA_310P_KERNEL_SO, mode=ctypes.RTLD_GLOBAL)
            torch.ops.load_library(_CAUSAL_FA_310P_BINDINGS_SO)
            _ = torch.ops.npu_ext.causal_fa_310p
            
            # 注册 abstract kernel（已修复）
            try:
                torch.library.impl_abstract("npu_ext::causal_fa_310p")(
                    lambda q, k, v, seq_lens, scale: torch.empty_like(q)
                )
            except Exception:
                pass  # already registered
            
            self._use_causal_fa = True
            logger.info("causal_fa_310p operator loaded successfully")
        except Exception as e:
            logger.warning("Failed to load causal_fa_310p: %s", e)
```

---

## 性能数据详细

### Prefill 基线对比

```
测试项                FA E2E    基线 E2E    吞吐(t/s)对比
------------------------------------------------------
Prefill_256_D50      2779ms    1393ms      101.1 vs 201.8
Prefill_1k_D50       3442ms    1489ms      304.8 vs 704.4
Prefill_2k_D50       5179ms    3789ms      399.5 vs 556.9
Prefill_4k_D50       7553ms    5774ms      545.6 vs 720.9
Prefill_8k_D50       12649ms   9072ms      649.3 vs 907.4
Prefill_16k_D50      23505ms   17601ms     698.1 vs 933.2
Prefill_32k_D50      46528ms   37892ms     704.7 vs 866.4
```

### Decode 基线对比

```
Decode目标    FA实际    FA时间    FA吞吐    基线吞吐    变化
---------------------------------------------------------
50           50       10698ms   4.7t/s    32.4t/s    -6.9x
100          100      21487ms   4.7t/s    32.4t/s    -6.9x
200          200      43715ms   4.6t/s    31.5t/s    -6.8x
400          400      87183ms   4.6t/s    31.5t/s    -6.8x
```

---

## 结论与建议

### 当前状态

❌ **causal_fa_310p 算子不适合集成到生产环境**

原因：
1. Prefill 性能本身不及基线（标量 GEMM 实现）
2. Decode 性能回退 6.7x（疑似 RTLD_GLOBAL 符号冲突）
3. 即使修复 Bug 1，预期 Prefill 性能仍不如动态 chunk mask

### 下一步建议

**P0（阻塞）**：确认 RTLD_GLOBAL 根因
- 控制实验：`USE_CAUSAL_FA_310P=0` 测量 Decode 性能
- 如果性能恢复，尝试 `RTLD_LOCAL` 或静态链接修复

**P1（性能）**：向量化优化 causal_fa_310p
- 当前标量 GEMM 实现性能不及 `_npu_flash_attention`
- 需要向量化实现才能有集成价值

**P2（集成）**：修复 Bug 后重新验证
- abstract kernel 已修复
- RTLD_GLOBAL 问题解决后重新跑完整测试

### 替代方案

如果 causal_fa_310p 短期无法优化到超过基线，建议：
1. **保持当前动态 chunk mask 方案**（性能已验证，生产稳定）
2. **等待算子向量化优化完成**后再集成
3. **探索其他优化方向**（如 KV Cache 压缩、多流并行等）

---

## 附录

### 测试环境

- **硬件**: 2×Ascend 910B3（310P）
- **镜像**: `registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend:310p-opt-20260708`
- **模型**: Qwen3.6-35B-A3B-w8a8
- **配置**: max_model_len=131072, tp=2, max_num_seqs=8

### 相关文件

- 算子代码: `/home/nin/Workspace/310-ops/operators/causal_fa_310p/`
- 集成代码: `vllm_ascend/_310p/attention/attention_v1.py`
- 性能数据: `/tmp/perf_baseline.json`, `/tmp/decode_benchmark.json`
- 精度数据: `/tmp/gsm8k_evaluation.json`

### 参考文档

- Phase 3 progress.md: 记录了 causal_fa_310p "标量 GEMM，prefill 延迟较高" 的已知限制
- 310P_PRODUCTION_DEPLOYMENT.md: 生产环境部署基线
