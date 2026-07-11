# 310P 优化项目 - 二轮开发状态同步

**同步日期**: 2026-07-09  
**目标受众**: 算子开发会话、并行优化会话等其他任务会话  
**目的**: 同步二轮开发完成情况，避免信息孤岛

---

## 一、仓库结构变更 ⚠️ 重要

### 1.1 从 llm-service 迁移到独立 fork

**旧结构（一轮 POC）**:
```
llm-service/
└── patches/310p-long-context/
    ├── metadata_builder.py    # 外挂 patch
    ├── attention_v1.py         # 外挂 patch
    └── attention_mask.py       # 外挂 patch
```

**新结构（二轮生产）**:
```
310p-vllm-ascend/                        # 独立 fork 仓库
├── vllm_ascend/_310p/attention/         # ✅ 源码烘焙
│   ├── metadata_builder.py
│   ├── attention_v1.py
│   └── attention_mask.py
├── Dockerfile.310p                      # Ubuntu 镜像
├── Dockerfile.310p.openEuler            # openEuler 镜像
├── .gitmodules                          # catlass submodule
└── development/docs/                    # 开发文档
```

**关键变化**:
1. ✅ **源码烘焙**: patch 代码直接在 vllm-ascend fork 中，不再外挂
2. ✅ **独立构建**: Dockerfile 直接构建，容器启动无需手动 cp patch
3. ✅ **版本控制**: git 管理完整代码，支持 CI/CD
4. ✅ **GDN 集成**: git submodules 初始化 catlass

### 1.2 仓库信息

| 项目 | 位置 |
|------|------|
| **主仓库** | `/home/nin/Workspace/310p-vllm-ascend` |
| **分支** | `feat/310p-opt` |
| **远程** | fork from `vllm-project/vllm-ascend` |
| **llm-service** | `/home/nin/Workspace/worktrees/llm-service/feat-qwen35b-moe-310p-optimization-20260520` |
| **算子开发** | `/home/nin/Workspace/310-ops/operators/causal_fa_310p/` |

---

## 二、核心技术方案

### 2.1 动态 Chunk Mask（已生产部署）

**问题**: O(L²) attention mask 预分配
- 128K: 65536² × 2B = 8 GB（触发 OOM）

**方案**: 动态生成 chunk mask
- Prefill: T × T（T ≤ 2048）→ 最大 8 MB
- Chunked Prefill: 2048 × 131072 → 256 MB（临时分配）
- **内存节省**: -99.9%

**代码位置**:
```python
# vllm_ascend/_310p/attention/metadata_builder.py
def build(...) -> AscendAttentionMetadata310:
    attn_mask = None  # 跳过预分配

# vllm_ascend/_310p/attention/attention_v1.py
def forward_prefill_310(...):
    T = aligned_tokens  # chunk 大小
    chunk_mask = torch.zeros((T, T), dtype=torch.float16, device=query.device)
    chunk_mask.masked_fill_(~tril, float("-inf"))
```

### 2.2 FlashAttention 算子（三轮优化方向）

**状态**: ✅ Phase 3 Mmad 完成，准备集成

**进展**:
- ✅ 精度验证：6/6 全部 PASS（fp16-matmul 模式）
- ✅ 编译产物：`build/libcausal_fa_kernel.so`
- ✅ **历史 hang bug（seq_len≥2048）已解决**
- ⏳ 下一步：容器集成测试

**位置**: `/home/nin/Workspace/310-ops/operators/causal_fa_310p/`

**预期收益**: 消除 O(L²) mask，进一步提升 prefill 吞吐

---

## 三、镜像构建完成

### 3.1 生产镜像

| 镜像 | 大小 | 状态 |
|------|------|------|
| `registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend:310p-opt-20260708` (Ubuntu) | 16 GB | ✅ 生产就绪 |
| `registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend:310p-opt-openeuler-20260708` | 16.7 GB | ✅ 生产就绪 |
| `registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend:310p-opt-openeuler-20260709` | 16.7 GB | ✅ 最新版本 |

**镜像特性**:
- ✅ 源码改动已烘焙（metadata_builder.py、attention_v1.py、attention_mask.py）
- ✅ GDN kernel 已编译
- ✅ 无需容器启动时手动 patch 和重编译
- ✅ 支持 128K 上下文（max_model_len=131072）

### 3.2 部署参数

**关键参数**:
```bash
--max-model-len 131072           # 128K 上下文
--max-num-batched-tokens 1024    # ⚠️ 硬约束（310P ATB 算子限制）
--reasoning-parser qwen3         # 思考过程解析
-tp 2                            # 双卡 Tensor Parallel
```

**部署指南**: `310p-vllm-ascend/development/docs/guides/310P_PRODUCTION_DEPLOYMENT.md`

---

## 四、Gateway 层部署

### 4.1 llm-service Gateway

**位置**: `/home/nin/Workspace/worktrees/llm-service/feat-qwen35b-moe-310p-optimization-20260520`

**配置**:
```yaml
# configs/backends.310p-128k.yaml
backends:
  vllm-310p-128k:
    type: openai_compatible
    api_base: http://127.0.0.1:18082/v1
    thinking_mode: chat_template_kwargs
    default_enable_thinking: false  # 默认关闭 thinking
    models:
      - name: qwen3.6-128k
        upstream_name: qwen3.6-128k-8c
```

**功能**:
- ✅ Thinking 默认关闭，按需开启（`enable_thinking: true`）
- ✅ 屏蔽底层 reasoning parser 复杂性
- ✅ 统一入口（端口 8001）

**启动**:
```bash
PORT=8001 BACKENDS_CONFIG_PATH=configs/backends.310p-128k.yaml \
.venv/bin/python main.py
```

---

## 五、验证数据

### 5.1 长上下文验证

| 测试场景 | Prompt Tokens | 结果 | 说明 |
|---------|--------------|------|------|
| 基础推理 | 11 | ✅ | content="1 + 1 = 2" |
| 长文本 | 40,018 | ✅ | 正常处理 |
| 极长文本 | 60,015 | ✅ | 正常处理 |
| 理论上限 | 131,072 | ✅ | 128K 配置生效 |

### 5.2 Gateway Thinking 控制

```bash
# 默认关闭 thinking
curl http://localhost:8001/v1/chat/completions \
  -d '{"model": "qwen3.6-128k", "messages": [...], "max_tokens": 100}'
# 响应: content="1 + 1 = 2", reasoning=null

# 显式开启 thinking
curl http://localhost:8001/v1/chat/completions \
  -d '{"model": "qwen3.6-128k", "messages": [...], "max_tokens": 300, "enable_thinking": true}'
# 响应: content="\n\n1 + 1 = 2", reasoning_len=692
```

### 5.3 精度基线

- GSM8K: 98%（一轮 POC 数据，二轮未重测）
- LongBench: 14.02（一轮 POC 数据）
- GPQA-Diamond: 80%（20样本，thinking 模式）

---

## 六、与一轮 POC 的差异

| 对比项 | 一轮 POC | 二轮生产 |
|--------|---------|---------|
| **部署方式** | 容器启动时手动 cp patch + 重编译 | 镜像已包含，直接启动 |
| **代码位置** | `llm-service/patches/` 外挂 | `310p-vllm-ascend/vllm_ascend/_310p/` 源码中 |
| **仓库结构** | llm-service 仓库中维护 patch | 310p-vllm-ascend 独立 fork |
| **镜像标签** | `nightly-main-310p`（官方 nightly） | `310p-opt-20260708`（自建） |
| **Gateway** | 无 | ✅ llm-service Gateway（thinking 控制） |
| **OOM 修复** | patch 外挂 | ✅ 源码烘焙 |
| **GDN 集成** | 未集成 | ✅ git submodules（catlass） |

---

## 七、当前服务状态

| 服务器 | 架构 | 镜像 | 容器 | 端口 | 状态 |
|--------|------|------|------|------|------|
| ails-a1 | aarch64 | `310p-opt-20260708` | vllm-310p-opt-128k-8c | 18082 | ✅ 运行中 |
| - | - | - | llm-gateway | 8001 | ✅ 运行中 |

---

## 八、三轮规划（prefill/并发优化）

### 8.1 核心目标

**提升 prefill 和并发吞吐量**，解决当前性能瓶颈

### 8.2 候选技术方向

**Prefill 优化**:
1. ⭐ **FlashAttention 算子集成**（Phase 3 完成，优先级 P0）
2. 权重预取（Prefetch）
3. 多流并行（Prefill/Decode overlap）

**并发优化**:
1. Continuous Batching 调优
2. KV Cache 优化
3. NPU 多流调度

**系统级优化**:
1. 图模式编译（torch.compile / GE graph）
2. SuperKernel 算子融合

### 8.3 第一步行动

**FlashAttention 算子容器集成测试**（建议优先）:
1. 将 `build/libcausal_fa_kernel.so` 集成到镜像
2. 修改 `attention_v1.py` 调用算子
3. 端到端验证精度和性能
4. 对比动态 chunk mask vs FlashAttention 性能

**详细规划**: `310p-vllm-ascend/docs/PHASE_3_PLANNING.md`

---

## 九、文档导航

### 9.1 核心文档

| 文档 | 路径 | 用途 |
|------|------|------|
| **二轮完成总结** | `310p-vllm-ascend/docs/PHASE_2_COMPLETION_SUMMARY.md` | 完整技术方案、性能数据、交付清单 |
| **三轮规划** | `310p-vllm-ascend/docs/PHASE_3_PLANNING.md` | prefill/并发优化路线图 |
| **部署指南** | `310p-vllm-ascend/development/docs/guides/310P_PRODUCTION_DEPLOYMENT.md` | 生产部署步骤 |
| **开发进度** | `310p-vllm-ascend/development/docs/development/progress.md` | 完整开发历程 |

### 9.2 算子相关文档

| 文档 | 状态 | 说明 |
|------|------|------|
| `development/docs/development/2.1-causal-fa-310p-development-completion.md` | ✅ Phase 3 完成 | 算子开发总结 |
| `development/docs/development/2.2-causal-fa-310p-vllm-integration.md` | ⏳ 准备集成 | 集成方案设计 |

---

## 十、关键约束与注意事项

### 10.1 硬约束

⚠️ **`--max-num-batched-tokens` 必须 ≤ 1024**
- 原因：310P ATB 算子在 2048 时精度异常
- 影响：chunked prefill batch 大小受限
- 解决方向：FlashAttention 算子可能绕过此限制

### 10.2 已知问题

**Reasoning Parser**:
- `--reasoning-parser qwen3` + `max_tokens` 过小 → content 为 null
- 解决：使用 Gateway（默认关闭 thinking）或增加 max_tokens

**多轮对话**:
- 多轮对话第二轮可能返回 null（nightly 镜像遗留问题）
- 状态：文档中记录，待定位

---

## 十一、同步要点（TL;DR）

**算子开发会话需要知道**:
1. ✅ FlashAttention 算子（causal_fa_310p）Phase 3 完成，hang bug 已解决
2. ✅ 生产镜像已就绪，可以开始容器集成测试
3. ✅ 仓库迁移到 `310p-vllm-ascend` fork，源码烘焙
4. ⏳ 三轮优化第一步：FlashAttention 算子集成验证

**并行优化会话需要知道**:
1. ✅ 二轮基础设施就绪（镜像、Gateway、128K 支持）
2. ✅ 当前瓶颈：prefill 性能、并发能力
3. ✅ 三轮规划已制定，等待 profiling 数据
4. ⚠️ max_num_batched_tokens ≤ 1024 硬约束

**其他会话需要知道**:
1. ✅ 代码位置变更：从 `llm-service/patches/` 迁移到 `310p-vllm-ascend/vllm_ascend/_310p/`
2. ✅ 部署方式变更：不再需要手动 cp patch，镜像开箱即用
3. ✅ Gateway 层已部署，thinking 控制正常
4. ✅ 文档体系完整，见 `310p-vllm-ascend/development/docs/README.md`

---

**最后更新**: 2026-07-09  
**维护者**: 310P 优化项目主会话  
**反馈**: 如有疑问或发现信息过时，请及时同步
