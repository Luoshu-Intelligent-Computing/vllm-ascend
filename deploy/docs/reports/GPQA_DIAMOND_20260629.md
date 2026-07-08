# GPQA-Diamond 精度测试报告

**测试日期**: 2026-06-29  
**服务**: vllm-qwen36-128k（max_model_len=131072，max_num_batched_tokens=1024）  
**数据集**: GPQA-Diamond（198题，本次取前20题）  
**评估指标**: MCQ 准确率（A/B/C/D 选项）  
**推理模式**: thinking 开启（enable_thinking=True），CoT 链式推理  
**max_out_len**: 15000 tokens

---

## 测试结果

| 指标 | 数值 |
|------|------|
| 测试样本数 | 20 |
| 正确数 | **16** |
| **准确率** | **80.0%** |
| 平均输出长度 | ~13,600 chars（含完整推理过程）|

---

## 参考对比

| 模型 | GPQA-Diamond | 备注 |
|------|-------------|------|
| GPT-4o | ~53% | OpenAI 官方 |
| Claude 3.5 Sonnet | ~65% | Anthropic 官方 |
| Gemini 1.5 Pro | ~46% | Google 官方 |
| Qwen3-32B（稠密） | ~65-70% | 官方报告 |
| **Qwen3.6-35B-A3B-w8a8（本次）** | **80%** | 20样本，仅供参考 |

⚠️ **注意**：20 样本量偏小，可能存在偶然误差。建议后续跑全量 198 题确认。

---

## 错误样本分析（4 题）

| 题号 | 正确答案 | 模型输出 | 输出长度 |
|------|---------|---------|---------|
| 1 | C | D | 16,792 chars |
| 12 | D | A | 25,904 chars（最长，推理最复杂）|
| 17 | C | D | 9,341 chars |
| 18 | B | A | 20,621 chars |

题目 12 和 18 的输出最长，说明模型做了大量推理但仍答错，属于真实知识边界问题。

---

## 技术验证

- ✅ 推理流程正常：thinking 内容隔离到 `reasoning` 字段，`content` 包含完整 CoT + 最终答案
- ✅ 答案格式正确：全部 20 题均以 `Answer: X` 结尾
- ✅ 服务稳定：全程无超时、无 OOM，decode 速度稳定 ~31 t/s
- ✅ max_out_len=15000 充足：最长输出约 25,904 chars，未触及上限

---

## 测试配置

```
模型配置文件: vllm_api_qwen36_128k_gpqa.py
  max_out_len: 15000
  temperature: 0.0
  enable_thinking: True（默认）

数据集: GPQA-Diamond（gpqa_diamond.csv，198题）
评估: GPQA_Simple_Eval_postprocess（提取最后一行 Answer: X）

Prompt 模板（openai_simple_eval 标准）:
  Answer the following multiple choice question.
  The last line of your response should be of the following format: 'Answer: $LETTER'
  Think step by step before answering.
```

---

**测试人**: CANNBot model-infer-optimize  
**日期**: 2026-06-29
