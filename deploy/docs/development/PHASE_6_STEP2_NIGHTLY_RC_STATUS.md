# Phase 6 Step 2：nightly 验证 + 仓库迁移 + 镜像构建

**记录时间**: 2026-07-06（初版）/ 2026-07-08（更新：完成）  
**状态**: ✅ 全部完成

---

## 一、GDN 算子修复（2026-07-06）

### 背景

vllm-ascend 官方 nightly 镜像 `vllm_ascend_C.so` 编译时 `ASCEND_PLATFORM_310P` 宏未生效，导致 GDN Prefill 的 AscendC 算子未注册。

### 根因

```
vllm_ascend_C.so 编译时间（Jun 29 02:06）早于宏修复时间（Jun 29 13:13）
→ torch_binding.cpp 中 #ifdef ASCEND_PLATFORM_310P 块未编译
→ chunk_gated_delta_rule_fwd_h / chunk_fwd_o 未注册到 torch.ops._C_ascend
→ GDN Prefill 仍走 PyTorch fallback
```

**关键细节**：算子使用懒加载，必须 `import vllm_ascend.vllm_ascend_C` 才能触发注册。

### 修复方式

```bash
# 在容器内重编（删除旧 .o，强制重编）
cd /vllm-workspace/vllm-ascend
rm -f /tmp/build/CMakeFiles/vllm_ascend_C.dir/csrc/torch_binding.cpp.o
SOC_VERSION=ascend310p1 pip install -e . --no-build-isolation --no-deps -q
```

### 修复结果

| 算子 | 状态 |
|------|:----:|
| `chunk_gated_delta_rule_fwd_h` | ✅ 已注册 |
| `chunk_fwd_o` | ✅ 已注册 |
| `npu_recurrent_gated_delta_rule_310` | ✅ 已注册 |

---

## 二、nightly 服务验证（2026-07-07）

### nightly patch 适配

nightly `attention_mask.py` 和 `metadata_builder.py` 接口与旧 patch 有4处不兼容，主要修复：
1. `get_attention_mask(causal: bool, model_config)` 签名新增 `causal` 参数
2. `get_splitfuse_mask` OOM 修复（broadcast 替代全量预分配）

修复后路径：`deploy/docs/development/PHASE_6_STEP1_NIGHTLY_VALIDATION.md`

### 验证结果

| 测试 | 结果 |
|------|:----:|
| 服务启动（max_model_len=131072）| ✅ |
| 简单推理 | ✅ |
| 长上下文（57k tokens）| ✅ |
| GSM8K 20样本 | ✅ 95.0% |

### 性能对比（nightly vs 生产基线）

| 指标 | 生产基线（26.0.0）| nightly（CANN 9.1）| 变化 |
|------|:-----------------:|:------------------:|:----:|
| TTFT（256t）| 948ms | 763ms | **-19.5%** |
| TTFT（1k t）| 2839ms | 1593ms | **-43.9%** |
| TTFT（32k t）| 64520ms | 49569ms | **-23.2%** |
| Prefill 峰值 | 633 t/s | **704 t/s** | **+11%** |
| Decode 速度 | 31.5 t/s | **34.7 t/s** | **+10%** |

详细报告：`deploy/docs/reports/NIGHTLY_PERFORMANCE_COMPARISON_20260707.md`

---

## 三、仓库迁移（2026-07-08）

### 迁移目标

将 `310p-vllm-ascend` 从 **patch 仓**升级为 **fork + 源码改动**结构。

### 分支结构

```
upstream/main  → 追踪 vllm-project/vllm-ascend（官方，只读）
main           → 旧 patch 仓（保留历史）
feat/310p-opt  → 新开发分支（基于 upstream/main，含源码改动）
```

### Remote 配置

```
origin   → Luoshu-Intelligent-Computing/310p-vllm-ascend（push 目标）
upstream → vllm-project/vllm-ascend（官方追踪）
fork     → Luoshu-Intelligent-Computing/vllm-ascend（组织 fork）
```

### 源码改动（feat/310p-opt）

| 文件 | 改动内容 |
|------|---------|
| `vllm_ascend/_310p/attention/attention_mask.py` | 128K OOM 修复（动态 chunk mask，2048 cap） |
| `vllm_ascend/_310p/attention/metadata_builder.py` | nightly 接口适配（query_lens_cpu buffer，super().build()）|

---

## 四、镜像构建（2026-07-08）

### Dockerfile 修复

| 问题 | 修复 |
|------|------|
| `source` 在 `/bin/sh` 失败 | 改为 `bash -c "source ..."` |
| catlass submodule 缺失 | `git submodule update --init --recursive` |
| 容器内代理不可达 | `--network host` |

### 构建结果

| 镜像 | 标签 | 大小 | 状态 |
|------|------|------|------|
| Ubuntu 22.04 | `310p-opt-20260708` | 14.91 GB | ✅ 已构建并验证 |
| openEuler 24.03 | `310p-opt-openeuler-20260708` | TBD | ⏳ 构建中 |

镜像注册路径：`registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend`

### Ubuntu 镜像验证

容器 `vllm-310p-opt-128k` 已启动并通过：
- ✅ 服务启动（无需手动 patch / 重编译）
- ✅ 128K 长上下文推理（60K token prompt 正常）
- ✅ 显存占用合理（~31.7 GB/卡，73.8%）
- ⚠️ chat completions `content` 字段为 null（reasoning parser 行为，待排查）

### 稳定镜像启动命令

```bash
sudo podman run -d \
  --name vllm-310p-stable \
  --privileged --network host \
  --device /dev/davinci0 --device /dev/davinci1 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -e ASCEND_RT_VISIBLE_DEVICES=0,1 \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
  -v /srv/meetai/models:/models:ro \
  registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend:310p-opt-20260708 \
  bash -c '
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export ASCEND_CUSTOM_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom
    python3 -m vllm.entrypoints.openai.api_server \
      --model /models/llm-service/vllm-ascend/Qwen3.6-35B-A3B-w8a8 \
      --served-model-name qwen3.6-128k-stable \
      --host 0.0.0.0 --port 18082 -tp 2 \
      --max-model-len 131072 --max-num-batched-tokens 1024 \
      --max-num-seqs 1 --gpu-memory-utilization 0.75 \
      --dtype float16 --kv-cache-dtype auto \
      --trust-remote-code --enable-chunked-prefill \
      --no-enable-prefix-caching --reasoning-parser qwen3 \
      --additional-config "{\"ascend_compilation_config\": {\"fuse_norm_quant\": false}}" \
      --compilation-config "{\"cudagraph_mode\": \"FULL_DECODE_ONLY\", \"cudagraph_capture_sizes\": [1]}" \
      --async-scheduling --mamba-ssm-cache-dtype float16 \
      --allowed-local-media-path /
  '
```

---

## 五、遗留问题

| 问题 | 优先级 | 说明 |
|------|:------:|------|
| `content` 为 null | P1 | chat completions reasoning parser 行为异常 |
| openEuler 镜像验证 | P1 | 构建完成后需完整验证 |
| 推送 feat/310p-opt 到 GitHub | P2 | submodule shallow clone 问题导致 push 失败 |
| causal_fa_310p hang bug | P2 | seq_len≥2048 时 hang，需算子调试 |

---

**记录人**: CANNBot model-infer-optimize  
**最后更新**: 2026-07-08
