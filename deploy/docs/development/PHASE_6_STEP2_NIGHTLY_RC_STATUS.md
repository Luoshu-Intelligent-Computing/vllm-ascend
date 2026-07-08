# Phase 6 Step 2：Nightly / RC 镜像 GDN 算子修复进展

**记录时间**: 2026-07-06  
**相关任务**: 周计划 T1（310P 模型推理性能优化）

---

## 一、背景

vllm-ascend 官方 nightly 和 v0.22.1rc1 镜像中，`vllm_ascend_C.so` 编译时 `ASCEND_PLATFORM_310P` 宏未生效，导致 GDN Prefill 的 AscendC 算子（`chunk_gated_delta_rule_fwd_h`、`chunk_fwd_o`）未编译进 Python 绑定，仍走 PyTorch fallback。

---

## 二、当前状态

### 2.1 ails-a1（aarch64）— nightly-main-310p

| 项目 | 状态 | 说明 |
|------|:----:|------|
| 容器 `vllm-nightly-build` | ✅ 运行中 | `quay.io/ascend/vllm-ascend:nightly-main-310p`，sleep infinity |
| vllm_ascend_C.so 重编译 | ✅ 完成 | `SOC_VERSION=ascend310p1`，编译约 30 分钟 |
| `chunk_gated_delta_rule_fwd_h` | ✅ 已注册 | `torch.ops._C_ascend` 可用 |
| `chunk_fwd_o` | ✅ 已注册 | 同上 |
| `npu_recurrent_gated_delta_rule_310` | ✅ 已注册 | 同上 |
| commit 为新镜像 | ❌ 未完成 | 修复仅在容器内，未持久化 |
| 启动 nightly 测试服务 | ❌ 未完成 | 尚未用 nightly 跑性能对比 |

**验证命令**（容器内）：
```bash
sudo podman exec vllm-nightly-build bash -c "
  source /usr/local/Ascend/cann-9.1.0-beta.1/set_env.sh
  export ASCEND_RT_VISIBLE_DEVICES=0,1
  export ASCEND_CUSTOM_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom
  python3 -c \"
import torch, torch_npu, vllm_ascend
import vllm_ascend.vllm_ascend_C
ops = torch.ops._C_ascend
print('chunk_gated_delta_rule_fwd_h:', hasattr(ops, 'chunk_gated_delta_rule_fwd_h'))
print('chunk_fwd_o:', hasattr(ops, 'chunk_fwd_o'))
\"
"
```

---

### 2.2 ails-a2（x86_64）— v0.22.1rc1-310p-openeuler

| 项目 | 状态 | 说明 |
|------|:----:|------|
| 原始镜像 | ✅ 存在 | `quay.io/ascend/vllm-ascend:v0.22.1rc1-310p-openeuler` |
| triton 依赖冲突修复 | ✅ 完成 | 删除残留空 triton 目录，torch_npu 正常导入 |
| 修复后镜像 | ✅ 已 commit | `vllm-ascend-310p-gdn:v0.22.1rc1-fixed` |
| vllm_ascend_C.so 重编译 | ❌ 未完成 | 因代理被中断，编译未执行 |
| `chunk_gated_delta_rule_fwd_h` | ❌ 未注册 | 需重编才能使用 |
| `chunk_fwd_o` | ❌ 未注册 | 同上 |

---

## 三、根因分析

两个镜像存在相同问题：

```
vllm_ascend_C.so 编译时间早于 ASCEND_PLATFORM_310P 宏修复时间
→ torch_binding.cpp 中 #ifdef ASCEND_PLATFORM_310P 块未编译
→ chunk_gated_delta_rule_fwd_h / chunk_fwd_o 未注册到 torch.ops._C_ascend
→ GDN Prefill 仍走 torch_chunk_gated_delta_rule PyTorch fallback
```

**已验证的修复方法**（ails-a1 nightly 成功）：

```bash
# 1. 进入容器
# 2. 设置 SOC_VERSION
source /usr/local/Ascend/cann-9.1.0-beta.1/set_env.sh
export SOC_VERSION=ascend310p1

# 3. 删除旧 object files（强制重编，避免 cmake 缓存）
cd /vllm-workspace/vllm-ascend
python3 setup.py build_ext --inplace --build-temp /tmp/vllm_build_310p 2>&1

# 4. 如果 make 跳过了编译（缓存），手动删除 .o
rm -f /tmp/vllm_build_310p/CMakeFiles/vllm_ascend_C.dir/csrc/torch_binding.cpp.o
make -j4 vllm_ascend_C

# 5. 安装
cp /tmp/vllm_build_310p/vllm_ascend_C*.so vllm_ascend/

# 6. 验证（必须显式触发懒加载）
python3 -c "
import torch, torch_npu, vllm_ascend
import vllm_ascend.vllm_ascend_C   # 触发懒加载，缺少这行算子不注册
print(hasattr(torch.ops._C_ascend, 'chunk_gated_delta_rule_fwd_h'))  # True
"
```

**关键细节**：
- `pip install -e .` 会尝试安装 triton-ascend 依赖失败，必须用 `setup.py build_ext`
- 编译约 20-30 分钟（包含 ascend_protobuf、abseil-cpp 第三方库）
- 用 `strings` 验证算子是否在 .so 中：`strings vllm_ascend_C*.so | grep chunk_gated_delta_rule_fwd_h`
- 算子使用懒加载，必须 `import vllm_ascend.vllm_ascend_C` 才能触发注册

---

## 四、下一步行动

### 立即可执行

| 步骤 | 操作 | 时间 |
|------|------|------|
| **T1-1** | ails-a1：commit vllm-nightly-build 为新镜像 | 5 分钟 |
| **T1-2** | ails-a1：用新 nightly 镜像启动测试服务（端口 18083，注入 patches）| 10 分钟 |
| **T1-3** | ails-a1：性能对比测试（Prefill/Decode vs 当前 633 t/s 基线）| 1-2 小时 |
| **T1-4** | ails-a2：v0.22.1rc1 镜像重编译（同 ails-a1 步骤）| 30 分钟 |

### 验收标准

- [ ] nightly 镜像下 `chunk_gated_delta_rule_fwd_h` 可调用
- [ ] nightly 测试服务（128K）正常启动
- [ ] Prefill 吞吐对比报告（目标：GDN 加速 ≥20%，即 ≥760 t/s）
- [ ] Decode 速度无回退（±5% 内）
- [ ] GSM8K 回归 ≥96%

---

## 五、参考信息

| 资源 | 路径/说明 |
|------|-----------|
| nightly 编译容器 | ails-a1 `vllm-nightly-build`（sleep infinity，可直接进入） |
| v0.22.1rc1 修复镜像 | ails-a2 `vllm-ascend-310p-gdn:v0.22.1rc1-fixed`（triton 已修复）|
| 当前生产基线 | ails-a1 `vllm-qwen36-128k`（Prefill ~633 t/s，Decode ~31.5 t/s）|
| 性能基线报告 | `docs/reports/BASELINE_REPORT_20260629.md` |
| Phase 6 总体计划 | `docs/development/PHASE_6_UPSTREAM_ALIGNMENT_PLAN.md` |

---

**状态**: 待推进（T1-1 → T1-2 → T1-3）  
**记录人**: CANNBot model-infer-optimize
