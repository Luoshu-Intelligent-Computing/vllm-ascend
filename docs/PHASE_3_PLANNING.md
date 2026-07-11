# 310P 优化项目 - 三轮规划

**规划时间**: 2026-07-09  
**目标**: 提升 prefill 和并发吞吐量，解决性能瓶颈  
**前置**: 二轮已完成 128K OOM 修复 + GDN 架构集成

---

## 一、当前状态

### 1.1 二轮成果
- ✅ 128K 长上下文支持（动态 chunk mask 解决 OOM）
- ✅ 镜像构建完成（Ubuntu + openEuler）
- ✅ Gateway 层部署（thinking 控制）
- ✅ 基础功能验证通过

### 1.2 已知瓶颈

根据用户反馈，当前存在以下性能瓶颈：
1. **Prefill 阶段性能不足**
2. **并发能力受限**
3. **整体吞吐率需要提升**

> **待补充**: 需要采集详细的性能基线数据来量化瓶颈

---

## 二、三轮目标

### 2.1 核心目标

**提升 prefill 和并发吞吐量**，使 310P 平台在长上下文场景下达到生产级性能要求。

### 2.2 量化指标（待确定）

**Prefill 性能**:
- 目标吞吐: TBD tokens/s（不同 batch size）
- 目标延迟: TBD ms（P50/P90/P99）

**并发性能**:
- 目标并发数: TBD requests
- 目标总吞吐: TBD tokens/s

**约束条件**:
- max_num_batched_tokens ≤ 1024（310P 硬约束）
- max_model_len = 131072（128K 上下文）

---

## 三、技术路线

### 3.1 第一阶段：性能诊断（Week 1）

#### 目标
建立性能基线，定位具体瓶颈

#### 任务清单
- [ ] **Profiling 采集**
  - [ ] msprof 采集 prefill/decode 算子级性能
  - [ ] 分析 Memory-bound vs Compute-bound
  - [ ] 识别热点算子（TOP 10）
  
- [ ] **基线测试**
  - [ ] 单请求 prefill 吞吐（batch_size=1, 不同 prompt 长度）
  - [ ] 多请求并发吞吐（不同并发数）
  - [ ] KV Cache 利用率分析
  
- [ ] **瓶颈定位**
  - [ ] Prefill 瓶颈：算子？内存？通信？
  - [ ] 并发瓶颈：调度？Cache？资源争抢？

#### 交付物
- 性能基线报告（含 profiling 数据）
- 瓶颈分析报告（定量+定性）
- 优化方案建议（按优先级排序）

### 3.2 第二阶段：方案设计（Week 2）

#### 候选优化技术

**A. Prefill 优化**
1. **FlashAttention 融合算子** ⭐ **最新进展**
   - ✅ **Phase 3 Mmad 路径修复完成**（2026-07-09）
   - ✅ 精度验证：6/6 全部 PASS（fp16-matmul 模式）
   - ✅ 编译产物：`build/libcausal_fa_kernel.so`
   - ⏳ **下一步**：容器集成测试
   - **位置**: `/home/nin/Workspace/310-ops/operators/causal_fa_310p/`
   - **预期收益**: 消除 O(L²) mask，提升 prefill 吞吐
   - **历史 hang bug（seq_len≥2048）已解决**

2. **权重预取（Prefetch）**
   - 评估 `torch_npu.npu_prefetch` 收益
   - 识别 Memory-bound 的 MatMul/QBMM/GMM
   - 设计预取策略

3. **多流并行**
   - Prefill/Decode 流水线重叠
   - 多模块并行（Attention/FFN/MoE）
   - TorchAir 多流改造

**B. 并发优化**
1. **Continuous Batching 调优**
   - 调整 `--max-num-seqs`
   - 优化 scheduling 策略
   - Dynamic batching 参数

2. **KV Cache 优化**
   - Paged Attention 参数调优
   - Block size 优化
   - Cache 淘汰策略

3. **资源调度**
   - NPU 多流调度
   - Tensor Parallel 通信优化
   - HCCL 参数调优

**C. 系统级优化**
1. **图模式编译**
   - torch.compile 评估
   - GE graph 模式
   - cudagraph 优化

2. **算子融合**
   - SuperKernel 二进制融合
   - 自定义融合算子
   - norm_quant / act_quant 融合

#### 任务
- [ ] 对每个候选技术进行可行性评估
- [ ] 设计实施方案（工作量、风险、预期收益）
- [ ] 确定优先级（quick win vs 长期收益）

#### 交付物
- 优化方案设计文档
- 实施计划（分阶段 Roadmap）

### 3.3 第三阶段：实施与验证（Week 3-6）

#### 实施策略
按优先级逐个实施，每个优化点独立验证：
1. 实施优化
2. 性能回归测试
3. 对比基线数据
4. 记录收益和问题

#### 验证标准
- 性能提升 ≥ X%（待定）
- 精度验证通过（GSM8K / LongBench）
- 稳定性测试通过（长时间运行无 crash）

#### 回滚策略
每个优化点保留独立分支，出现问题可快速回滚

---

## 四、风险与依赖

### 4.1 技术风险
- **310P 平台限制**: 某些优化技术可能不支持
- **精度退化**: 激进优化可能影响模型精度
- **稳定性**: 新特性可能引入 runtime 错误

### 4.2 资源依赖
- **硬件**: 需要 310P 测试环境（已有）
- **模型**: Qwen3.6-35B-A3B-w8a8（已有）
- **工具**: msprof、torch_npu profiler（已有）

### 4.3 时间约束
- 建议预留缓冲时间应对意外问题
- 关键路径：profiling → 瓶颈定位 → 方案设计

---

## 五、成功标准

### 5.1 必达目标
- [ ] 完成性能基线采集和瓶颈定位
- [ ] 至少实施 3 个优化点
- [ ] Prefill 吞吐提升 ≥ X%（待定）
- [ ] 并发吞吐提升 ≥ Y%（待定）

### 5.2 理想目标
- [ ] 实施 5+ 个优化点
- [ ] Prefill 吞吐提升 ≥ 2X
- [ ] 并发吞吐提升 ≥ 3X
- [ ] 形成完整的优化方法论文档

---

## 六、下一步行动

### 6.1 立即启动
1. **环境准备**: 确认 msprof 工具可用
2. **测试脚本**: 准备性能测试脚本（不同 prompt 长度、并发数）
3. **Profiling 采集**: 开始采集基线性能数据

### 6.2 待确认
- [ ] 三轮的具体时间安排（开始/结束日期）
- [ ] 量化的性能目标（吞吐/延迟指标）
- [ ] 优先级排序（哪些优化点优先实施）

---

## 七、相关文档

- [二轮完成总结](PHASE_2_COMPLETION_SUMMARY.md)
- [msprof Profiling 指南](../patches/310p-long-context/docs/development/PROFILING_GUIDE.md)（待创建）
- [优化实施记录](../patches/310p-long-context/docs/development/PHASE_3_OPTIMIZATION_LOG.md)（待创建）

---

**备注**: 本文档是三轮优化的初步规划框架，具体实施细节将在第一阶段（性能诊断）完成后补充完善。
