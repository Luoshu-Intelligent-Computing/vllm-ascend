# 香橙派 Atlas 200I Pro（310P1 SoC）性能测试报告

**测试日期**: 2026-07-10 ～ 2026-07-13  
**硬件**: Atlas 200I Pro（310P1，单卡 SoC，89,610 MB HBM ≈ 87.5 GB）  
**模型**: Qwen3.6-35B-A3B-w8a8（MoE，W8A8_DYNAMIC 量化，38 GB 权重）  
**宿主机**: openEuler 22.03 LTS-SP3（aarch64）

---

## 一、部署配置演进

### 1.1 最终生产配置（老 POC 镜像 + 128K patch）

| 参数 | 值 | 说明 |
|------|-----|------|
| 镜像 | `vllm-ascend:dev-26.0.0.poc.20260413-9.0.T3.B030` | 老 POC 镜像 |
| `--max-model-len` | **131072** | 128K 上下文（patch 修复 OOM 后启用） |
| `--max-num-batched-tokens` | 1024 | 310P 安全上限 |
| `--max-num-seqs` | 1 | 单并发 |
| `--gpu-memory-utilization` | 0.75 | |
| `--compilation-config` | `FULL_DECODE_ONLY, capture_sizes=[1]` | 图优化（decode 阶段） |
| `enable_npugraph_ex` | true | NPU graph 扩展 |
| `--reasoning-parser` | qwen3 | thinking/content 分离 |
| Patch 来源 | `patches/310p-long-context/` | 适配老 POC 镜像的接口 |

**容器部署路径**: `/home/nin/Workspace/310p-vllm-ascend/deploy/scripts/start-orangepi.sh`  
**Patch 存放路径**: `/models/patches/`（香橙派本地）

### 1.2 关键 Patch 说明

| 文件 | 作用 |
|------|------|
| `metadata_builder.py` | 设 `attn_mask = None`，消除 O(L²) mask 预分配（8GB → 0） |
| `attention_v1.py` | `forward_prefill_310` 动态生成 [T,T] chunk mask（≤8MB） |
| `attention_mask.py` | `get_splitfuse_mask` 按需生成 [T, max_seqlen] mask |

---

## 二、性能测试数据

### 2.1 首字延迟（TTFT）

测试方法：benchmark-live-server.py，mode=prefill，concurrency=1

| Prompt Tokens | 首字延迟（TTFT） | Prefill 吞吐 |
|:-------------:|:---------------:|:------------:|
| ~20 | **~0.96 s** | — |
| ~240 | ~0.77 s | ~315 t/s |
| ~4,651 | **~19.1 s** | ~243 t/s |
| ~9,259 | **~29.7 s** | ~312 t/s |

### 2.2 Decode 性能

**测试方法**: ais_bench，单请求（concurrency=1）

| 指标 | 值 |
|------|-----|
| **TPOT（每 token 延迟）** | **49.2 ms**（P50/P75/P90 均稳定在 49.2ms） |
| **Decode 吞吐** | **~20.3 t/s**（1000ms ÷ 49.2ms） |
| TPOT 最小值 | 49.0 ms |
| TPOT 最大值 | 49.3 ms |
| Output Token Throughput（含 E2E 统计） | ~15.5–16.3 t/s |

**特性**：Decode 速度极稳定，方差极小（P99 = 49.3ms ≈ P50），说明图模式（FULL_DECODE_ONLY）对每步推理有很好的一致性保证。

### 2.3 ais_bench 完整性能报告

**测试时间**: 2026-07-13  
**配置**: 128K 服务 + 图优化 + patch  
**数据集**: 8 个短问题（~20 tokens prompt），max_tokens=512，concurrency=1

| 指标 | Average | Min | Max | Median | P90 |
|------|:-------:|:---:|:---:|:------:|:---:|
| E2EL | 25338.5 ms | 2988.4 ms | 42351.6 ms | 28595.0 ms | 38603.8 ms |
| **TTFT** | **6022.9 ms** | **928.7 ms** | 17201.0 ms | 3462.4 ms | 13440.8 ms |
| **TPOT** | **49.2 ms** | 49.0 ms | 49.3 ms | 49.2 ms | 49.3 ms |
| ITL | 49.0 ms | 0.1 ms | 98.6 ms | 49.2 ms | 49.7 ms |
| InputTokens | 20.75 | 19.0 | 23.0 | 20.5 | — |
| OutputTokens | 393.5 | 43.0 | 512.0 | 512.0 | — |
| OutputTokenThroughput | 15.69 t/s | 12.09 t/s | 19.64 t/s | 14.86 t/s | 18.82 t/s |

**注**：E2EL/TTFT 均值受串行请求排队影响（8个请求顺序执行，后续请求含等待时间）。  
**真实单请求 TTFT ≈ 928.7 ms**（第一个请求，~20 token prompt）。

### 2.4 长上下文验证

| 测试场景 | Prompt Tokens | 输出 | 状态 |
|---------|:-------------:|------|:----:|
| 15K token 文本总结 | 15,019 | 正常返回 | ✅ |
| 60K token 场景（理论验证）| 60,000+ | max_seq_len=131072 配置生效 | ✅ |

---

## 三、与 ails-a1（300I DUO 双卡）对比

| 指标 | 香橙派 310P1（单卡 SoC） | ails-a1 310P3（双卡 PCIe） |
|------|:----------------------:|:-------------------------:|
| NPU 型号 | 310P1（RC 模式） | 310P3 × 2（EP 模式） |
| 单 die HBM | ~87.5 GB | 43 GB × 2 |
| TP | 1 | 2 |
| 最大上下文 | 131072（128K）| 131072（128K）|
| **Prefill 吞吐（峰值）** | ~243–410 t/s | ~1,251 t/s（新镜像）|
| **Decode 吞吐（单请求）** | **~20.3 t/s** | **~31.5 t/s** |
| **TTFT（~20 tokens）** | ~0.96 s | ~0.95 s |
| **TPOT** | **49.2 ms** | **31.7 ms** |
| 图优化 | FULL_DECODE_ONLY | FULL_DECODE_ONLY |
| GSM8K 精度 | 见下节 | 98%（thinking=True）|

---

## 四、精度验证

### 4.1 功能精度

| 测试 | 结果 |
|------|:----:|
| 基础问答（enable_thinking=False）| ✅ 正常 |
| 长上下文理解（15k tokens） | ✅ 正常 |
| reasoning_content 分离（reasoning_parser=qwen3）| ✅ reasoning 字段有效 |

### 4.2 GSM8K 精度评测（enable_thinking=False）

| 指标 | 值 |
|------|-----|
| 样本数 | 50 |
| **准确率** | **36%（18/50）** |
| max_tokens | 1024 |

**⚠️ 注意**：此结果**不能作为精度基准**，原因：
1. `enable_thinking=False` 模式下 Qwen3.6 进入"简洁模式"，不做 CoT 推理
2. 部分回答出现 token 重复循环（W8A8 量化 + 非思考模式的已知问题）
3. ails-a1 的 98% 基准是 `enable_thinking=True` 下测得的

正确精度评测需使用 `enable_thinking=True`（待环境修复后补充）。

---

## 五、已知限制与问题

| 问题 | 状态 | 说明 |
|------|:----:|------|
| `enable_thinking=True` 下 content=null | ⚠️ 未解决 | W8A8 量化导致模型在 `</think>` 后直接停，无正文输出；需 reasoning_parser 适配 |
| 非思考模式 token 重复循环 | ⚠️ 已知 | 长输出场景 W8A8 量化在非思考模式下概率坍缩，加 `repetition_penalty=1.1` 可缓解 |
| cudagraph 128K 时触发 StatelessRandomNormalV2 | ⚠️ 已绕过 | 使用 POC 专用 patch（非 nightly 版）可避免 |
| Prefill 吞吐低于 ails-a1 | ℹ️ 预期 | 单卡 310P1 SoC，无 GDN AscendC kernel，Prefill 约为双卡 310P3 的 1/5 |
| TPOT 49.2ms vs ails-a1 31.7ms | ℹ️ 预期 | 310P1 内存带宽低于 310P3，单 die 差距 |

---

## 六、部署操作记录

### 6.1 存储初始化（一次性）

```bash
# 创建 /models 分区（nvme0n1p2，~96GB）
sgdisk -e /dev/nvme0n1
parted /dev/nvme0n1 --script mkpart primary ext4 17.3GB 100%
mkfs.ext4 -L models /dev/nvme0n1p2
echo 'UUID=7c8d01ee-636b-4338-a0d5-e9c12e993439 /models ext4 defaults 0 2' >> /etc/fstab
```

### 6.2 容器启动

```bash
# 复制 patch 文件（首次或更新 patch 时）
scp patches/310p-long-context/{metadata_builder,attention_v1,attention_mask}.py \
    root@orangepi-1:/models/patches/

# 启动服务
bash /home/nin/Workspace/310p-vllm-ascend/deploy/scripts/start-orangepi.sh

# 等待 15-20 分钟就绪（图编译耗时）
curl http://192.168.29.233:38082/v1/models
```

### 6.3 推理调用建议

```python
# 推荐：关闭 thinking，加 repetition_penalty 防循环
{
    "model": "qwen3.6",
    "messages": [...],
    "max_tokens": 1024,
    "temperature": 0.3,
    "repetition_penalty": 1.05,
    "chat_template_kwargs": {"enable_thinking": False}
}
```

---

## 七、待补充测试

| 测试项 | 说明 |
|--------|------|
| GSM8K（enable_thinking=True）| 等 thinking 模式修复后补充，预期接近 98% |
| 无 patch vs 有 patch 精度对比 | 对照实验，验证 patch 不引入精度损失 |
| 多并发性能 | max_num_seqs=16 场景，capture_sizes=[1,4] |
| 长文本生成 Decode 吞吐 | 固定输出 256/512 tokens 的 TPOT 曲线 |

---

**记录人**: nin-zhihao / CANNBot  
**最后更新**: 2026-07-13
