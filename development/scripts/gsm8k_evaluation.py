#!/usr/bin/env python3
"""
GSM8K 精度评估脚本（并发版）
使用 vllm 服务评估数学推理能力，支持多并发加速
"""
import json, re, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request, urllib.error
from datasets import load_from_disk

API_URL = "http://localhost:18082/v1/chat/completions"
MODEL = "qwen3.6-128k-8c"
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

def evaluate_gsm8k(num_samples=None, concurrency=4):
    """评估 GSM8K（并发版）"""
    print(f"=== GSM8K 精度评估 ===")
    print(f"数据集: {GSM8K_PATH}")
    print(f"API: {API_URL}")
    print(f"并发数: {concurrency}\n")

    dataset = load_from_disk(GSM8K_PATH)
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    total_samples = len(dataset)
    print(f"测试样本数: {total_samples}\n")

    # 线程安全计数
    lock = threading.Lock()
    correct = 0
    total = 0
    errors = 0
    completed = 0

    def evaluate_one(idx, item):
        question = item['question']
        gt_answer = item['answer'].split('####')[-1].strip()
        gt_normalized = normalize_answer(gt_answer)

        messages = [
            {"role": "system", "content": "你是一个数学专家。请详细解答以下数学问题，最后用 #### 后跟答案的数字。"},
            {"role": "user", "content": question}
        ]
        result = call_api(messages, max_tokens=1024, temperature=0.0)
        return idx, gt_normalized, result

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(evaluate_one, i, item): i
                   for i, item in enumerate(dataset)}

        for future in as_completed(futures):
            nonlocal_vars = {}
            try:
                idx, gt_normalized, result = future.result()
            except Exception as e:
                with lock:
                    errors += 1
                    completed += 1
                    print(f"[{completed}/{total_samples}] ERROR: {e}", flush=True)
                continue

            with lock:
                total += 1
                completed += 1

                if "error" in result:
                    errors += 1
                    print(f"[{completed}/{total_samples}] ERROR: {result['error']}", flush=True)
                else:
                    pred_text = result['choices'][0]['message'].get('content', '')
                    pred_answer = extract_answer(pred_text)
                    pred_normalized = normalize_answer(pred_answer)
                    is_correct = (pred_normalized == gt_normalized)

                    if is_correct:
                        correct += 1

                    status = "✓" if is_correct else "✗"
                    print(f"[{completed}/{total_samples}] {status}  GT={gt_normalized}  Pred={pred_normalized}", flush=True)

                    if not is_correct and pred_text:
                        print(f"  模型输出: {pred_text[:150]}...", flush=True)

                if completed % 10 == 0:
                    acc = correct / total * 100 if total > 0 else 0
                    print(f"  进度: {completed}/{total_samples}  准确率: {acc:.1f}% ({correct}/{total})\n", flush=True)

    print("\n=== 评估结果 ===")
    print(f"总样本数: {total_samples}")
    print(f"成功评估: {total}")
    print(f"错误/超时: {errors}")
    print(f"正确数: {correct}")
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"准确率: {accuracy:.2f}%")

    result_path = "/tmp/gsm8k_evaluation.json"
    with open(result_path, "w") as f:
        json.dump({
            "dataset": "GSM8K",
            "total_samples": total_samples,
            "evaluated": total,
            "errors": errors,
            "correct": correct,
            "accuracy": accuracy,
            "concurrency": concurrency,
            "model": MODEL,
            "api_url": API_URL,
        }, f, indent=2)
    print(f"\n结果已保存: {result_path}")
    return accuracy

if __name__ == "__main__":
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    concurrency = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    accuracy = evaluate_gsm8k(num_samples=num_samples, concurrency=concurrency)
    sys.exit(0 if accuracy > 0 else 1)
