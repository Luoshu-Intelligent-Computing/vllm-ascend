#!/usr/bin/env python3
"""
GSM8K 精度评估脚本
使用 vllm 64k 服务评估数学推理能力
"""
import json, re, sys
from pathlib import Path
import urllib.request, urllib.error
from datasets import load_from_disk

API_URL = "http://localhost:18082/v1/chat/completions"
MODEL = "qwen3.6-128k"
GSM8K_PATH = "/home/nin/Workspace/gsm8k_data/test"

def call_api(messages, max_tokens=512, temperature=0.0):
    """调用 vLLM API"""
    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False}
    }).encode("utf-8")
    req = urllib.request.Request(API_URL, data=payload,
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}

def extract_answer(text):
    """从模型输出中提取数值答案"""
    # 尝试多种模式：#### 123, answer: 123, **123**, 最后一个数字等
    patterns = [
        r'####\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',  # GSM8K 标准格式
        r'answer\s*(?:is|:)?\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',
        r'\*\*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\*\*',
        r'(?:^|\n)\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:$|\n)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).replace(',', '')

    # Fallback: 最后一个数字
    numbers = re.findall(r'([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', text)
    if numbers:
        return numbers[-1].replace(',', '')
    return None

def normalize_answer(ans):
    """标准化答案格式"""
    if ans is None:
        return None
    ans = str(ans).strip().replace(',', '')
    try:
        # 尝试转为浮点数再转回字符串（去除前导零等）
        return str(float(ans))
    except:
        return ans

def evaluate_gsm8k(num_samples=None):
    """评估 GSM8K"""
    print(f"=== GSM8K 精度评估 ===")
    print(f"数据集: {GSM8K_PATH}")
    print(f"API: {API_URL}\n")

    # 加载数据集
    dataset = load_from_disk(GSM8K_PATH)
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    print(f"测试样本数: {len(dataset)}\n")

    correct = 0
    total = 0
    errors = 0

    for i, item in enumerate(dataset):
        question = item['question']
        # GSM8K answer 格式："#### 123"
        gt_answer = item['answer'].split('####')[-1].strip()
        gt_normalized = normalize_answer(gt_answer)

        messages = [
            {"role": "system", "content": "你是一个数学专家。请详细解答以下数学问题，最后用 #### 后跟答案的数字。"},
            {"role": "user", "content": question}
        ]

        result = call_api(messages, max_tokens=1024, temperature=0.0)

        if "error" in result:
            print(f"[{i+1}/{len(dataset)}] ERROR: {result['error']}")
            errors += 1
            continue

        pred_text = result['choices'][0]['message'].get('content', '')
        pred_answer = extract_answer(pred_text)
        pred_normalized = normalize_answer(pred_answer)

        is_correct = (pred_normalized == gt_normalized)
        if is_correct:
            correct += 1

        total += 1

        status = "✓" if is_correct else "✗"
        print(f"[{i+1}/{len(dataset)}] {status}  GT={gt_normalized}  Pred={pred_normalized}", flush=True)

        # 每题都输出进度，避免看起来卡死
        if not is_correct and pred_text:
            print(f"  模型输出: {pred_text[:150]}...", flush=True)

        if (i + 1) % 10 == 0:
            acc = correct / total * 100 if total > 0 else 0
            print(f"  进度: {i+1}/{len(dataset)}  准确率: {acc:.1f}% ({correct}/{total})\n", flush=True)

    # 最终统计
    print("\n=== 评估结果 ===")
    print(f"总样本数: {len(dataset)}")
    print(f"成功评估: {total}")
    print(f"错误/超时: {errors}")
    print(f"正确数: {correct}")
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"准确率: {accuracy:.2f}%")

    # 保存结果
    result_path = "/tmp/gsm8k_evaluation.json"
    with open(result_path, "w") as f:
        json.dump({
            "dataset": "GSM8K",
            "total_samples": len(dataset),
            "evaluated": total,
            "errors": errors,
            "correct": correct,
            "accuracy": accuracy,
            "model": MODEL,
            "api_url": API_URL,
        }, f, indent=2)
    print(f"\n结果已保存: {result_path}")

    return accuracy

if __name__ == "__main__":
    # 先测试 50 个样本，如需全集评估可去掉 num_samples 参数
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    accuracy = evaluate_gsm8k(num_samples=num_samples)
    sys.exit(0 if accuracy > 0 else 1)
