#!/usr/bin/env python3
"""
GSM8K 精度评估 - OrangePi 310P 服务
验证 KV Cache 精度修复效果
"""
import json, re, sys, time
from datetime import datetime
from pathlib import Path
import urllib.request
import pyarrow as pa

API_URL   = "http://orangepi-1:38081/v1/chat/completions"
MODEL     = "qwen3.6"
DATA_PATH = "/home/nin/Workspace/gsm8k_data/test/data-00000-of-00001.arrow"
OUT_DIR   = Path("/tmp/gsm8k_orangepi")

SYSTEM_PROMPT = (
    "你是一个数学专家。请一步步推理解答以下数学题，"
    "最后在单独一行写出 #### 后跟答案数字（仅数字，不加单位）。"
)

def call_api(question: str, max_tokens: int = 512) -> dict:
    payload = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        API_URL, data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())

def extract_answer(text: str):
    for pat in [
        r'####\s*([+-]?\d[\d,]*(?:\.\d+)?)',
        r'answer\s*(?:is|=|:)\s*([+-]?\d[\d,]*(?:\.\d+)?)',
        r'\*\*([+-]?\d[\d,]*(?:\.\d+)?)\*\*',
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).replace(',', '')
    nums = re.findall(r'[+-]?\d[\d,]*(?:\.\d+)?', text)
    return nums[-1].replace(',', '') if nums else None

def normalize(ans):
    if ans is None:
        return None
    try:
        v = float(str(ans).replace(',', ''))
        return str(int(v)) if v == int(v) else str(v)
    except:
        return str(ans).strip()

def main():
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_consec_fail = 5

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fail_file = OUT_DIR / f"failures_{ts}.jsonl"
    summary_file = OUT_DIR / f"summary_{ts}.json"

    with pa.ipc.open_stream(DATA_PATH) as f:
        table = f.read_all()
    total_avail = len(table)
    n = min(num_samples, total_avail)
    print(f"GSM8K 评估  |  样本: {n}/{total_avail}  |  服务: {API_URL}")
    print(f"失败案例 -> {fail_file}\n")

    correct = consec_fail = errors = 0
    failures = []

    for i in range(n):
        q   = table['question'][i].as_py()
        ans = table['answer'][i].as_py()
        gt  = normalize(ans.split('####')[-1].strip())

        t0 = time.time()
        try:
            resp = call_api(q)
            pred_text = resp['choices'][0]['message']['content']
            pred = normalize(extract_answer(pred_text))
            elapsed = time.time() - t0
        except Exception as e:
            errors += 1
            consec_fail += 1
            print(f"[{i+1:4d}] ERR  {e}")
            if consec_fail >= max_consec_fail:
                print(f"\n连续失败 {max_consec_fail} 次，停止评估。")
                break
            continue

        ok = (pred == gt)
        if ok:
            correct += 1
            consec_fail = 0
            mark = "✓"
        else:
            consec_fail += 1
            mark = "✗"
            entry = {
                "idx": i + 1, "question": q,
                "ground_truth": gt, "predicted": pred,
                "model_output": pred_text,
            }
            failures.append(entry)
            with open(fail_file, 'a') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        done = i + 1 - errors
        acc = correct / done * 100 if done else 0
        print(f"[{i+1:4d}] {mark}  GT={gt:<8} Pred={str(pred):<8} ({elapsed:.1f}s)", flush=True)

        if (i + 1) % 10 == 0:
            print(f"  ── 进度 {i+1}/{n}  准确率 {acc:.1f}%  ({correct}/{done})  连续失败={consec_fail}\n", flush=True)

        if consec_fail >= max_consec_fail:
            print(f"\n连续失败 {max_consec_fail} 次，停止评估。")
            break

    done = (i + 1) - errors
    acc  = correct / done * 100 if done else 0

    print("\n" + "="*50)
    print(f"评估完成")
    print(f"  已评估:  {done}")
    print(f"  正确:    {correct}")
    print(f"  错误:    {errors}")
    print(f"  准确率:  {acc:.2f}%")
    print(f"  失败案例: {len(failures)} 条 -> {fail_file}")

    summary = {
        "timestamp": ts, "model": MODEL, "api_url": API_URL,
        "num_requested": n, "evaluated": done, "errors": errors,
        "correct": correct, "accuracy": round(acc, 2),
        "failures_count": len(failures), "failures_file": str(fail_file),
    }
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  汇总:    {summary_file}")

    # 简单失败案例分析
    if failures:
        print("\n── 失败案例抽样分析 (最多5条) ──")
        for e in failures[:5]:
            print(f"\n题目 #{e['idx']}: {e['question'][:80]}...")
            print(f"  期望: {e['ground_truth']}")
            print(f"  预测: {e['predicted']}")
            out = e['model_output']
            print(f"  输出末尾: ...{out[-120:].strip()}")

if __name__ == '__main__':
    main()
