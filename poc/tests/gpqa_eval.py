#!/home/nin/Workspace/.venv/bin/python3
"""
GPQA Diamond 精度评估脚本
数据集：GPQA Diamond（198 samples，研究生级别科学问题）
评估指标：Accuracy（模型答案与正确答案匹配率）
"""
import json, time, random, re, urllib.request, urllib.error
from datasets import load_from_disk

API_URL = "http://localhost:18082/v1/chat/completions"
MODEL = "qwen3.6-64k"
DATASET_PATH = "/home/nin/Workspace/gpqa_data"

def call_api(messages, max_tokens=512, timeout=300):
    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "thinking": {"type": "enabled", "budget_tokens": 2048},
    }).encode("utf-8")
    req = urllib.request.Request(API_URL, data=payload,
                                  headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        r = json.loads(resp.read())
    elapsed = time.monotonic() - t0
    return r, elapsed

def build_prompt(item):
    """构造多选题 prompt，随机分配 A/B/C/D 标签"""
    question = item["Question"]
    correct = item["Correct Answer"]

    texts = [correct, item["Incorrect Answer 1"], item["Incorrect Answer 2"], item["Incorrect Answer 3"]]
    random.shuffle(texts)

    labels = ["A", "B", "C", "D"]
    correct_label = labels[texts.index(correct)]

    prompt = f"{question}\n\n"
    for label, text in zip(labels, texts):
        prompt += f"{label}. {text}\n"
    prompt += "\nPlease answer with only the letter (A, B, C, or D) of the correct option."

    return prompt, correct_label

def extract_answer(response_text):
    """从模型输出中提取答案标签（A/B/C/D）"""
    patterns = [
        r'^\s*([ABCD])\b',
        r'\b([ABCD])\s*$',
        r'answer\s+is\s+([ABCD])\b',
        r'correct\s+(?:answer|option)\s+is\s+([ABCD])\b',
        r'\b([ABCD])\s+is\s+correct',
    ]
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    match = re.search(r'\b([ABCD])\b', response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None

def main():
    print("=== GPQA Diamond 精度评估 ===")
    print(f"API: {API_URL}")
    print(f"Model: {MODEL}")
    print(f"Dataset: {DATASET_PATH}\n")

    # 等待服务
    for _ in range(12):
        try:
            urllib.request.urlopen("http://localhost:18082/v1/models", timeout=5)
            print("[INFO] 服务就绪\n")
            break
        except:
            time.sleep(5)
    else:
        print("[ERROR] 服务未就绪")
        return

    # 加载数据集
    print("[INFO] 加载 GPQA Diamond 数据集...")
    dataset = load_from_disk(DATASET_PATH)
    train_split = dataset["train"]
    print(f"[INFO] 加载完成，共 {len(train_split)} 个样本\n")

    # 评估所有样本
    results = []
    correct_count = 0

    for idx, item in enumerate(train_split):
        print(f"[{idx+1}/{len(train_split)}] Record ID: {item.get('Record ID', 'N/A')}", flush=True)

        # 构造 prompt
        prompt, correct_label, options = build_prompt(item)
        messages = [
            {"role": "system", "content": "You are a helpful assistant. For multiple choice questions, respond with ONLY the single letter of the correct answer (A, B, C, or D). No explanation, no reasoning, just the letter."},
            {"role": "user", "content": prompt}
        ]

        # 调用模型
        try:
            r, elapsed = call_api(messages, max_tokens=100)
            response_text = r["choices"][0]["message"].get("content") or ""
            predicted_label = extract_answer(response_text) if response_text else None
            is_correct = (predicted_label == correct_label)

            if is_correct:
                correct_count += 1

            result = {
                "idx": idx,
                "record_id": item.get("Record ID", "N/A"),
                "subdomain": item.get("Subdomain", "N/A"),
                "correct_label": correct_label,
                "predicted_label": predicted_label,
                "is_correct": is_correct,
                "response": response_text[:200],
                "elapsed_ms": round(elapsed * 1000, 1),
            }
            results.append(result)

            status = "✓" if is_correct else "✗"
            print(f"  Correct: {correct_label}, Predicted: {predicted_label}  {status}", flush=True)
            print(f"  Response: {response_text[:100]}", flush=True)
            print(f"  Time: {elapsed*1000:.0f}ms", flush=True)

            # 每 10 个样本输出一次进度汇总
            if (idx + 1) % 10 == 0:
                current_acc = correct_count / (idx + 1) * 100
                print(f"\n>>> 进度: {idx+1}/{len(train_split)} 完成，当前准确率: {current_acc:.2f}% ({correct_count}/{idx+1})\n", flush=True)
            else:
                print("", flush=True)

        except Exception as e:
            print(f"  ERROR: {e}\n", flush=True)
            results.append({
                "idx": idx,
                "record_id": item.get("Record ID", "N/A"),
                "error": str(e),
                "is_correct": False,
            })

    # 汇总统计
    accuracy = correct_count / len(train_split) * 100
    print("\n=== 评估结果 ===")
    print(f"总样本数: {len(train_split)}")
    print(f"正确数: {correct_count}")
    print(f"错误数: {len(train_split) - correct_count}")
    print(f"准确率: {accuracy:.2f}%")

    # 按子领域统计
    subdomain_stats = {}
    for r in results:
        if "error" in r:
            continue
        subdomain = r.get("subdomain", "Unknown")
        if subdomain not in subdomain_stats:
            subdomain_stats[subdomain] = {"total": 0, "correct": 0}
        subdomain_stats[subdomain]["total"] += 1
        if r["is_correct"]:
            subdomain_stats[subdomain]["correct"] += 1

    print("\n=== 按子领域统计 ===")
    for subdomain, stats in sorted(subdomain_stats.items()):
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"{subdomain}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")

    # 保存结果
    output_path = "/tmp/gpqa_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "summary": {
                "total": len(train_split),
                "correct": correct_count,
                "accuracy": accuracy,
            },
            "subdomain_stats": subdomain_stats,
            "details": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] 详细结果已写入 {output_path}")

if __name__ == "__main__":
    main()
