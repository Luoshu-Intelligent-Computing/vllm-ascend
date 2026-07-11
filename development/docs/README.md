# 310P 优化项目文档导航

**最后更新**: 2026-07-09  
**当前阶段**: 二轮开发完成

---

## 📋 文档索引

### 开发进度与总结
- **[progress.md](development/progress.md)** - 完整开发历程（一轮 POC → 二轮生产化）
- **[PHASE_6_STEP2_NIGHTLY_RC_STATUS.md](development/PHASE_6_STEP2_NIGHTLY_RC_STATUS.md)** - 二轮最终状态（2026-07-09）

### 部署指南
- **[310P_PRODUCTION_DEPLOYMENT.md](guides/310P_PRODUCTION_DEPLOYMENT.md)** - 生产部署方案（推荐）
- **[GATEWAY_DEPLOYMENT_GUIDE.md](guides/GATEWAY_DEPLOYMENT_GUIDE.md)** - Gateway 层部署

### 性能与精度报告
- **[NIGHTLY_PERFORMANCE_COMPARISON_20260707.md](reports/NIGHTLY_PERFORMANCE_COMPARISON_20260707.md)** - 性能对比
- **[CONCURRENCY_ANALYSIS_20260708.md](reports/CONCURRENCY_ANALYSIS_20260708.md)** - 并发分析
- **[GSM8K_COMPARISON_20260626.md](reports/GSM8K_COMPARISON_20260626.md)** - GSM8K 精度验证

### 历史记录（参考）
- **[2.1-causal-fa-310p-development-completion.md](development/2.1-causal-fa-310p-development-completion.md)** - causal_fa_310p 算子开发（已放弃）
- **[2.2-causal-fa-310p-vllm-integration.md](development/2.2-causal-fa-310p-vllm-integration.md)** - 集成方案（已放弃）

---

## ⚠️ 重要提示

### 当前生产方案
- ✅ **镜像**: `registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend:310p-opt-20260708`
- ✅ **技术路线**: 动态 chunk mask（源码烘焙）
- ✅ **Gateway**: llm-service Gateway 层（端口 8001）
- ❌ **已放弃**: causal_fa_310p AscendC kernel（seq_len≥2048 时 hang）

### 文档状态说明
1. **progress.md**: 包含完整历史，包括已放弃的 causal_fa_310p 路径
2. **causal_fa_310p 相关文档**: 仅作历史参考，该技术路线已放弃
3. **nightly patch 文档**: 描述旧的外挂 patch 方案，新镜像已源码烘焙

### 阅读建议
- **新用户**: 直接看 `guides/310P_PRODUCTION_DEPLOYMENT.md`
- **运维人员**: 重点看 `guides/` 目录
- **开发人员**: 参考 `progress.md` 了解技术演进

---

## 📊 二轮开发成果

### 核心指标
- **128K 上下文**: ✅ 支持（max_model_len=131072）
- **OOM 修复**: ✅ 8 GB → 8 MB (-99.9%)
- **镜像构建**: ✅ Ubuntu + openEuler 双版本
- **Gateway 部署**: ✅ Thinking 控制生产就绪

### 验证数据
- 60K tokens 长文本推理通过
- GSM8K 98% 精度
- Gateway thinking 控制正常

---

## 🔗 相关资源

- **二轮完成总结**: [../../docs/PHASE_2_COMPLETION_SUMMARY.md](../../docs/PHASE_2_COMPLETION_SUMMARY.md)
- **三轮规划**: [../../docs/PHASE_3_PLANNING.md](../../docs/PHASE_3_PLANNING.md)
- **仓库 README**: [../../README.md](../../README.md)

## 最新更新（2026-07-11）

### 性能报告
- [PERFORMANCE_REPORT_20260711_BATCHED4096.md](reports/PERFORMANCE_REPORT_20260711_BATCHED4096.md) - max_num_batched_tokens=4096 性能与精度验证报告
  - 精度：多长度 6/6，GSM8K 98%（49/50），无回退
  - 性能：中大规模（>2k）提升 +8~26%，小规模（~1.5k）下降 -35%
  - 决策：需根据业务 prompt 长度分布选择配置

### 测试脚本
- [development/scripts/gsm8k_evaluation.py](../scripts/gsm8k_evaluation.py) - GSM8K 并发评估脚本
  - 用法：`python3 gsm8k_evaluation.py <样本数> <并发数>`
  - 默认：50 样本，4 并发
  - 加速：约 4x vs 串行版本
