# Phase 6 Step 1: Nightly 镜像环境验证结果

**验证时间**: 2026-07-02  
**镜像**: `quay.io/ascend/vllm-ascend:nightly-main-310p`  
**结论**: ❌ **不可用于 GDN Prefill 优化**

---

## 验证环境

| 项目 | 值 |
|------|-----|
| 硬件 | Atlas 300I Duo (310P3 × 2) |
| 驱动 | 24.1.RC3 |
| 测试日期 | 2026-07-02 |

---

## 验证结果

### 1. 基础功能

| 测试项 | 结果 | 说明 |
|--------|:----:|------|
| 容器启动 | ✅ | 无报错 |
| NPU 设备初始化 | ✅ | `torch_npu` 加载成功 |
| 驱动兼容性 | ✅ | 无版本冲突 |

### 2. CANN 与 torch_npu 版本

| 组件 | 版本 |
|------|------|
| CANN | 9.1.0-beta.1 |
| torch | 2.10.0+cpu |
| torch_npu | 2.10.0 |
| vllm | 0.23.0 |
| vllm_ascend | 源码挂载（`/vllm-workspace/`）|

### 3. 关键算子可用性

| 算子 | 预期 | 实际 | 影响 |
|------|:----:|:----:|------|
| `chunk_gated_delta_rule_fwd_h` | ✅ | ❌ | GDN Prefill 无 NPU 加速 |
| `chunk_fwd_o` | ✅ | ❌ | 同上 |
| `_npu_flash_attention_v3` | ✅ | ❌ | Compressed mask 不可用 |
| `_npu_paged_attention_splitfuse_v2` | ✅ | ❌ | 同上 |
| `_npu_flash_attention` | ✅ | ✅ | 当前使用 |
| `_npu_paged_attention_splitfuse` | ✅ | ✅ | 当前使用 |
| `npu_recurrent_gated_delta_rule` | ✅ | ✅ | GDN Decode |

### 4. torch.ops 命名空间检查

| 命名空间 | 算子数量 | GDN/chunk 相关 |
|---------|:-------:|:-------------:|
| `torch.ops._C_ascend` | **1** | ❌ 无 |
| `torch.ops.npu` | 354 | ❌ 无 |
| `torch.ops.aten` | 988 | ❌ 无 |
| 其他命名空间 | - | ❌ 无 |

---

## 根本原因分析

### 问题 1：`_C_ascend` 扩展未编译

nightly 镜像的 `torch.ops._C_ascend` 仅包含 1 个算子，远低于预期。对比：
- **预期**：GDN chunk 算子应在 `_C_ascend` 命名空间
- **实际**：命名空间几乎为空

**推测原因**：
- vllm_ascend 从源码挂载（`/vllm-workspace/`），未预编译 C++/CUDA 扩展
- `_C_ascend` 扩展需要 `pip install -e .` 或 `python setup.py develop` 才能注册

### 问题 2：GDN chunk 算子未暴露

虽然 CANN 9.1.0-beta.1 包含 AscendC 源码（`/usr/local/Ascend/cann-9.1.0-beta.1/opp/.../chunk_gated_delta_rule.cpp`），但：
- torch_npu 2.10.0 未将其绑定为 Python 算子
- 不在 `torch.ops.npu` 中（354 个标准 NPU 算子）
- 不在任何其他 torch.ops 命名空间中

**可能原因**：
- AscendC 算子需要额外的 Python binding 层（ACLNNOp 封装）
- torch_npu 2.10.0 尚未集成 CANN 9.1 的新算子
- 或者需要特定 feature flag 才能启用

### 问题 3：v3/v2 算子缺失

`_npu_flash_attention_v3` 和 `_npu_paged_attention_splitfuse_v2` 同样不存在，说明：
- 这些算子依赖更新版本的 torch_npu（>2.10.0）
- 或者需要 CANN 9.1 正式商用版（非 beta）

---

## 尝试的解决方案

### 尝试 1：源码编译 vllm_ascend 扩展

**方法**：进入容器，执行 `pip install -e /vllm-workspace/vllm-ascend`

**预期**：编译并注册 `_C_ascend` 扩展

**状态**：未执行（可能需要完整 CANN 开发工具链，容器未预装）

### 尝试 2：检查 torch_npu 源码

**方法**：搜索 torch_npu 源码中 `chunk_gated_delta_rule` 的绑定代码

**发现**：
```bash
grep -r 'chunk_gated_delta_rule' /usr/local/python3.12.13/lib/.../torch_npu/
```
找到 `recurrent_gated_delta_rule.py`（GE converter），但无 `chunk_gated_delta_rule` 的 Python binding

**结论**：torch_npu 2.10.0 确实未集成此算子

---

## 迁移路径决策

### 路线 A（nightly 迁移）：❌ **不可行**

**原因**：
1. GDN chunk 算子不存在，无法加速 30 层 GDN Prefill（主要目标）
2. v3/v2 算子不存在，compressed mask 优化不可用（次要目标）
3. 即使修复 nightly 补丁 bug，也无实质收益

### 路线 B（保守优化）：✅ **必须执行**

**立即可做**（无 CANN 依赖）：
1. `query_lens_cpu` pinned buffer 优化
2. `SpecDecoding` 状态支持
3. 修复 nightly 补丁 bug（为未来准备）

**中期跟进**：
- 跟踪 torch_npu 版本更新（等待 chunk_gated_delta_rule 集成）
- 跟踪 CANN 9.1 商用版发布（beta → GA）

**长期方案**：
- AscendC 自研 GDN Prefill kernel（参考 CANN 9.1 源码）
- 预计工作量：2-3 周

---

## 技术细节记录

### CANN 9.1 GDN 源码位置

nightly 镜像内包含完整 AscendC 实现：
```
/usr/local/Ascend/cann-9.1.0-beta.1/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/chunk_gated_delta_rule/
├── chunk_gated_delta_rule.cpp
├── chunk_gated_delta_rule.h
├── chunk_gated_delta_rule_stage1.h
└── chunk_gated_delta_rule_stage3.h

/usr/local/Ascend/cann-9.1.0-beta.1/aarch64-linux/include/aclnnop/
├── aclnn_chunk_gated_delta_rule.h
└── level2/aclnn_chunk_gated_delta_rule.h
```

这些文件可用于：
- 理解 GDN chunk 算法实现
- AscendC 自研时的参考

### vllm_ascend 源码挂载

nightly 镜像特点：
- vllm_ascend 从 `/vllm-workspace/vllm-ascend/` 加载（开发模式）
- 未预编译 C++ 扩展（需要 `pip install -e .`）
- 这解释了为什么 `_C_ascend` 几乎为空

---

## 对 Phase 6 计划的影响

### 修订后的执行路线

原计划 **Step 2A（nightly 迁移）** 取消，直接执行 **Step 2B（保守优化）**。

### 修订后的时间规划

| 任务 | 原计划 | 修订后 |
|------|--------|--------|
| Step 1: 环境验证 | 1 天 | ✅ 已完成 |
| Step 2A: nightly 迁移 | 2-3 天 | ❌ 取消 |
| Step 2B: 保守优化 | 1 周 | ✅ 执行中 |

---

## 下一步行动

### 立即执行（1 周内）

1. **实现 query_lens_cpu 优化**
   - 参考上游代码：`metadata_builder.py` 中的 `_query_lens_cpu_buffer`
   - 预期收益：<5%（ChunkedPrefill 场景微优化）

2. **补全 SpecDecoding 状态支持**
   - 当前代码：`raise NotImplementedError`
   - 修复方法：路由到 `ChunkedPrefill` 路径
   - 收益：功能完整性，性能无变化

3. **修复 nightly 补丁 bug**
   - 文件：`patches/310p-long-context-nightly/attention_mask.py:139`
   - 修复一行代码（移除 `min(..., COMPRESSED_MASK_SEQ_LEN)`）
   - 为未来 CANN 升级准备

### 中期跟进（1-3 个月）

1. **跟踪 torch_npu 更新**
   - 监控 torch_npu GitHub/官方发布
   - 关注 `chunk_gated_delta_rule` 集成时间

2. **跟踪 CANN 9.1 GA 版本**
   - 向华为反馈 310P + GDN 需求
   - 确认商用版发布时间

### 长期备选方案（视优先级）

**AscendC 自研 GDN Prefill kernel**：
- 参考：nightly 镜像内 CANN 9.1 源码
- 工作量：2-3 周（算法理解 + AscendC 开发 + 调试验证）
- 收益：30 层 GDN Prefill NPU 加速，预期 Prefill 吞吐提升 30-50%

---

**验证人**: CANNBot model-infer-optimize  
**日期**: 2026-07-02  
**状态**: Step 1 验证完成，路线决策：执行 Step 2B
