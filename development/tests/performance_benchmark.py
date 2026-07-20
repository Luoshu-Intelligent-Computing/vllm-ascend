#!/usr/bin/env python3
"""
性能基线采集脚本
测试 Prefill TTFT、Decode 吞吐量、E2E 延迟
"""
import json, time, statistics, urllib.request, urllib.error, sys

API_URL = "http://localhost:18082/v1/chat/completions"
MODEL = "qwen3.6-128k"

def call_api(messages, max_tokens=100, timeout=600):
    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False}
    }).encode("utf-8")
    req = urllib.request.Request(API_URL, data=payload,
                                  headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        r = json.loads(resp.read())
    elapsed = time.monotonic() - t0
    return r, elapsed

def build_prompt(n_tokens_approx):
    # 每单元约 12 tokens
    unit = "巴黎是法国的首都，也是欧洲最重要的城市之一。"
    repeats = max(1, n_tokens_approx // 12)
    return unit * repeats + "\n\n请问本文提到的城市名字是什么？"

def run_test(label, prompt_tokens_target, decode_tokens, repeats=3):
    prompt = build_prompt(prompt_tokens_target)
    messages = [{"role": "user", "content": prompt}]
    results = []
    print(f"\n[{label}] prompt≈{prompt_tokens_target}t decode={decode_tokens}t ×{repeats}", flush=True)
    for i in range(repeats):
        try:
            r, elapsed = call_api(messages, max_tokens=decode_tokens)
            usage = r.get("usage", {})
            p_tokens = usage.get("prompt_tokens", 0)
            c_tokens = usage.get("completion_tokens", 0)
            finish = r["choices"][0]["finish_reason"]
            content = r["choices"][0]["message"].get("content") or ""
            # E2E latency
            e2e_ms = elapsed * 1000
            # Approx TTFT: prefill_tokens / prefill_throughput (from engine logs, not available here)
            # Use e2e_ms / total_tokens as proxy for decode-dominated requests
            results.append({
                "e2e_ms": e2e_ms,
                "prompt_tokens": p_tokens,
                "completion_tokens": c_tokens,
                "finish_reason": finish,
                "content_ok": bool(content and len(content) > 0),
            })
            print(f"  run{i+1}: {e2e_ms:.0f}ms  p={p_tokens} c={c_tokens} ok={bool(content)}", flush=True)
        except Exception as e:
            print(f"  run{i+1}: ERROR {e}", flush=True)
            results.append({"e2e_ms": None, "error": str(e)})

    valid = [r for r in results if r.get("e2e_ms") is not None]
    if not valid:
        return {"label": label, "status": "FAIL", "results": results}

    e2e_vals = [r["e2e_ms"] for r in valid]
    avg_e2e = statistics.mean(e2e_vals)
    p_tokens = valid[0]["prompt_tokens"]
    c_tokens = statistics.mean([r["completion_tokens"] for r in valid])

    # Prefill throughput = prompt_tokens / (e2e - decode_time)
    # Decode time approx = c_tokens * avg_decode_latency (unknown here, estimate from short prefix)
    # For reporting: just record raw e2e and token counts
    return {
        "label": label,
        "status": "PASS",
        "avg_e2e_ms": round(avg_e2e, 1),
        "prompt_tokens": p_tokens,
        "avg_completion_tokens": round(c_tokens, 1),
        "e2e_throughput_tokens_per_s": round((p_tokens + c_tokens) / (avg_e2e / 1000), 1),
        "results": results,
    }

def main():
    print("=== Qwen3.6-35B-A3B-w8a8 @ 310P×2  性能基线采集 ===")
    print(f"API: {API_URL}  Model: {MODEL}")
    print("注：enable_thinking=False，排除 reasoning 开销")

    # 等待服务就绪
    for _ in range(60):
        try:
            r = urllib.request.urlopen(f"http://localhost:18082/v1/models", timeout=5)
            print("\n[INFO] 服务就绪")
            break
        except:
            time.sleep(5)
    else:
        print("[ERROR] 服务未就绪，退出")
        sys.exit(1)

    tests = [
        # (label, prompt_tokens_approx, decode_tokens)
        ("Prefill_256_D50",    256,   50),
        ("Prefill_1k_D50",    1024,   50),
        ("Prefill_2k_D50",    2048,   50),
        ("Prefill_4k_D50",    4096,   50),
        ("Prefill_8k_D50",    8192,   50),
        ("Prefill_16k_D50",  16384,   50),
        ("Prefill_32k_D50",  32768,   50),
        ("Decode_256_D200",    256,  200),
        ("Decode_1k_D200",    1024,  200),
        ("Decode_4k_D200",    4096,  200),
    ]

    all_results = []
    for label, p, d in tests:
        result = run_test(label, p, d, repeats=2)
        all_results.append(result)

    # 输出汇总表
    print("\n\n=== 性能汇总 ===")
    print(f"{'测试项':<22} {'状态':<6} {'E2E(ms)':<10} {'Prompt':<8} {'Decode':<8} {'吞吐(t/s)':<12}")
    print("-" * 70)
    for r in all_results:
        if r["status"] == "PASS":
            print(f"{r['label']:<22} PASS   {r['avg_e2e_ms']:<10.0f} {r['prompt_tokens']:<8} {r['avg_completion_tokens']:<8.0f} {r['e2e_throughput_tokens_per_s']:<12.1f}")
        else:
            print(f"{r['label']:<22} FAIL")

    # 写入 JSON
    out_path = "/tmp/perf_baseline.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] 详细结果已写入 {out_path}")

if __name__ == "__main__":
    main()
