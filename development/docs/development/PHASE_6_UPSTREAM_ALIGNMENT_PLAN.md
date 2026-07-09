# Phase 6: 对齐官方最新代码优化计划

**启动时间**: 2026-07-02  
**目标**: 吸收 vllm-ascend 官方在 310P 上的最新优化特性，最大化性能  
**约束**: 不影响当前运行的 128K 服务（localhost:18082）

---

## 一、背景与动机

### 当前状态（2026-06-29 基线）

| 指标 | 数值 | 说明 |
|------|------|------|
| 128K 服务 | ✅ 生产可用 | GSM8K 98%, GPQA 80%, LongBench 14.02 |
| Prefill 峰值 | ~633 t/s（8K规模） | max_num_batched_tokens=1024 |
| Decode 速度 | ~31.5 t/s | 稳定 |
| 显存/卡 | ~31.7 GB | 73.8% 利用率 |

### 已知瓶颈

1. **GDN Prefill**（最大瓶颈）：30 层 GDN 走 `torch_chunk_gated_delta_rule` PyTorch fallback，无 NPU 加速
2. **Prefill 性能下降**：max_batched=2048→1024 导致吞吐下降 34%（960→633 t/s），是 ATB 算子精度约束的必要代价
3. **动态 mask 开销**：每步 prefill 动态生成 mask（`torch.ones` + `tril` + `nd_to_nz_2d`）

### 官方最新进展

- **CANN 9.1.0-beta.1**：nightly 镜像内置 GDN chunk AscendC kernel
- **Compressed Mask (v3/v2 算子)**：固定 2048×2048 预计算 mask，替代动态生成
- **GDN Prefill 优化**：`chunk_gated_delta_rule` AscendC 实现，理论可加速 30 层 GDN Prefill

---

## 二、调研发现

### 2.1 环境差异

| 项目 | 生产镜像（26.0.0） | nightly 镜像 | 影响 |
|------|:-----------------:|:------------:|------|
| CANN 版本 | 9.0.T3 | **9.1.0-beta.1** | nightly 含新算子 |
| `chunk_gated_delta_rule` kernel | ❌ 无 | ✅ **有**（AscendC）| GDN Prefill 可加速 |
| `_npu_flash_attention_v3` | ❌ 无 | 待验证 | compressed mask |
| `_npu_paged_attention_splitfuse_v2` | ❌ 无 | 待验证 | compressed mask |
| 驱动兼容性 | 24.1.RC3（当前）| 待验证 | 可能有兼容性风险 |

### 2.2 代码差异（当前 patch vs 官方上游）

| 特性 | 当前实现 | 官方方案 | CANN 依赖 | 收益评估 |
|------|---------|---------|----------|---------|
| **GDN Prefill** | PyTorch fallback | AscendC kernel | 9.1+ | **高**（30层主瓶颈）|
| **Compressed Mask** | 动态生成 [T,T] | 预计算 [2048,2048] + v3/v2算子 | 9.1+ ATB | 中（消除每步开销）|
| **query_lens_cpu** | `query_start_loc.cpu()` D2H | pinned buffer预分配 | 无 | 低（微优化）|
| **SpecDecoding** | `NotImplementedError` | 路由到 ChunkedPrefill | 无 | 低（功能完整性）|
| **metadata 图捕获支持** | 缺失 | splitfuse 图捕获优化 | 9.1+ | 中（图模式性能）|

### 2.3 nightly 补丁失败根因（已定位）

`patches/310p-long-context-nightly/attention_mask.py:139` 硬编码 bug：

```python
# 错误：无论是否支持压缩，都限制到 2048
max_seq_len = COMPRESSED_MASK_SEQ_LEN if self.support_compressed_mask \
              else min(self.max_seqlen, COMPRESSED_MASK_SEQ_LEN)  # ← BUG

# 正确：不支持压缩时应使用完整 max_seqlen
max_seq_len = COMPRESSED_MASK_SEQ_LEN if self.support_compressed_mask \
              else self.max_seqlen
```

---

## 三、技术路线决策树

```
START
  │
  ├─→ 验证 nightly 镜像在 310P3 + driver 24.1.RC3 上可用性
  │     │
  │     ├─ 可启动且 GDN kernel 可用 → 路线A（迁移到 nightly）
  │     │     ├─ 修复 nightly 补丁 bug（一行改动）
  │     │     ├─ 在测试容器验证 128K 服务
  │     │     ├─ 性能对比（GDN 加速 + compressed mask）
  │     │     └─ 精度对比（GSM8K/GPQA 回归测试）
  │     │
  │     └─ 不可用或 GDN kernel 不可用 → 路线B（保守优化）
  │           ├─ 立即可做：query_lens_cpu + SpecDecoding
  │           ├─ 中期：等待 CANN 9.1 商用版
  │           └─ 长期：AscendC 自研 GDN kernel（参考 9.1 源码）
  │
  END
```

---

## 四、Phase 6 执行计划

### Step 1: 环境验证（不影响生产）

**目标**: 确认 nightly 镜像（CANN 9.1）在当前硬件/驱动下可用性

**任务**:
1. 在非生产端口启动 nightly 容器（如 18083）
2. 验证 NPU 设备初始化、torch_npu 加载
3. 检查关键算子可用性：
   ```python
   hasattr(torch.ops._C_ascend, 'chunk_gated_delta_rule_fwd_h')
   hasattr(torch_npu, '_npu_flash_attention_v3')
   hasattr(torch_npu, '_npu_paged_attention_splitfuse_v2')
   ```
4. 测试短请求推理（14 tokens）验证功能正常

**预期时间**: 1-2 小时

**阻塞条件**:
- CANN 9.1 与 driver 24.1.RC3 不兼容
- nightly 镜像 310P 专属算子缺失

### Step 2A: nightly 可用时的迁移路径

**前提**: Step 1 验证通过

**任务**:
1. **修复 nightly 补丁 bug**（`attention_mask.py:139`）
2. **注入修复后的补丁到 nightly 容器**
3. **启动测试服务**（端口 18083，不影响生产 18082）
   - 配置：`max_model_len=131072, max_num_batched_tokens=1024`
4. **功能验证**:
   - 14→64k tokens 推理测试
   - GSM8K 50 样本快速回归（目标 ≥96%）
5. **性能对比**（vs 当前基线）:
   - Prefill 吞吐（期望提升，GDN 加速）
   - Decode 速度（应无变化）
   - TTFT（8K/16K/32K prompt）
6. **精度验证**:
   - GSM8K 50 样本（目标 98%）
   - GPQA 20 样本（目标 ≥75%）
7. **评估决策**:
   - 性能提升 ≥20% 且精度无回归 → 切换到 nightly
   - 否则 → 回退到路线 B

**预期时间**: 2-3 天

**交付物**:
- 性能对比报告（`NIGHTLY_MIGRATION_BENCHMARK.md`）
- 精度对比报告（`NIGHTLY_MIGRATION_ACCURACY.md`）
- 迁移决策文档

### Step 2B: nightly 不可用时的保守路径

**前提**: Step 1 验证失败

**任务**:
1. **立即优化**（无 CANN 依赖）:
   - 实现 `query_lens_cpu` pinned buffer（参考上游代码）
   - 补全 `SpecDecoding` 状态支持
   - 预期收益：微小（<5%）
2. **中期跟进**（等待 CANN 9.1 商用版发布）:
   - 向华为反馈 310P + 9.1 需求
   - 跟踪商用版发布时间
3. **长期方案**（若华为无时间表）:
   - AscendC 自研 GDN Prefill kernel
   - 参考 CANN 9.1 源码（nightly 镜像内 `/usr/local/Ascend/cann-9.1.0-beta.1/opp/.../chunk_gated_delta_rule.cpp`）
   - 预期工作量：2-3 周（包含调试验证）

**预期时间**: 立即优化 1 周，长期方案视优先级定

---

## 五、风险控制

### 5.1 环境风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| nightly 与 driver 不兼容 | 中 | 高（阻塞迁移）| Step 1 提前验证，准备回退路线 B |
| CANN 9.1 beta 稳定性问题 | 中 | 高（生产风险）| 充分测试，保留 9.0.T3 容器 |
| GDN kernel 310P 不可用 | 低 | 高（主要收益丢失）| 镜像内已有源码，确认编译目标 |

### 5.2 迁移风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| compressed mask 引入精度问题 | 低 | 高 | GSM8K/GPQA 回归测试，容忍度 ≤2% |
| 性能提升不明显（<10%） | 中 | 中 | 明确收益阈值（≥20%），否则不迁移 |
| 新算子在 128K 场景行为异常 | 低 | 高 | 边界测试（64K/128K extreme prompt）|

### 5.3 生产保护

**强制约束**:
1. **双容器并行**：nightly 测试服务（18083）与生产服务（18082）隔离
2. **回滚就绪**：保留当前稳定镜像和配置，可一键回退
3. **精度红线**：GSM8K <96% 或 GPQA <75% 立即终止迁移
4. **渐进切换**：先内部验证，再灰度流量，最后全量

---

## 六、成功标准

### 必须达成（Go/No-Go 判据）

- [ ] nightly 镜像在当前环境下可启动（NPU 初始化成功）
- [ ] GDN chunk kernel 或 v3/v2 算子至少一项可用
- [ ] 128K 长上下文功能验证通过（14→64K tokens）
- [ ] GSM8K ≥96%（允许 -2% 波动）
- [ ] GPQA ≥75%（允许 -5% 波动）

### 期望达成（性能目标）

- [ ] Prefill 吞吐提升 ≥20%（GDN 加速收益）
- [ ] Decode 速度无回退（±5% 内）
- [ ] TTFT 下降 ≥15%（32K prompt）

### 可选达成（附加收益）

- [ ] 图模式兼容性改善（compressed mask + metadata 优化）
- [ ] `max_num_batched_tokens` 上限突破（2048 精度问题在 9.1 中修复）

---

## 七、时间规划

| 里程碑 | 预计时间 | 输出物 |
|--------|---------|--------|
| Step 1: 环境验证 | 1 天 | 环境验证报告 |
| Step 2A: 迁移验证 | 2-3 天 | 性能/精度对比报告 + 决策文档 |
| Step 2B: 保守优化 | 1 周 | query_lens_cpu + SpecDecoding 补丁 |
| 生产切换（若 2A 成功）| 1 天 | 部署指南更新 |

**总计**: 4-7 天（取决于 nightly 可用性）

---

## 八、文档输出规范

所有阶段产出文档存放路径：

```
patches/310p-long-context/docs/
├── development/
│   ├── PHASE_6_UPSTREAM_ALIGNMENT_PLAN.md      # 本文档
│   ├── PHASE_6_STEP1_NIGHTLY_VALIDATION.md     # 环境验证结果
│   └── PHASE_6_STEP2_MIGRATION_DECISION.md     # 迁移决策
└── reports/
    ├── NIGHTLY_MIGRATION_BENCHMARK_20260702.md # 性能对比
    └── NIGHTLY_MIGRATION_ACCURACY_20260702.md  # 精度对比
```

---

## 九、关键参考

- **调研报告**: `/home/nin/Workspace/worktrees/.../vllm-ascend-310p-optimization-research-report.md`
- **Progress**: `docs/development/progress.md`（工作区融合算子调研章节）
- **当前基线**: `docs/reports/BASELINE_REPORT_20260629.md`
- **nightly 补丁**: `patches/310p-long-context-nightly/`（含已知 bug）
- **nightly 镜像**: `quay.io/ascend/vllm-ascend:nightly-main-310p`

---

**维护者**: CANNBot model-infer-optimize  
**创建时间**: 2026-07-02  
**状态**: 初始版本，待 Step 1 验证启动
