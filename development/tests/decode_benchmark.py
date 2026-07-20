#!/usr/bin/env python3
"""
Decode 专项性能测试：固定极短 prompt，长 decode，精确测量 decode 吞吐
方法：
  1. 先用极短 prompt 测一次 E2E_base（decode=1），得到纯 prefill 耗时基准
  2. 再用同 prompt 测 E2E_long（decode=N），两者相减得到 decode 总时间
  3. decode 吞吐 = N / decode_time
"""
import json, time, statistics, urllib.request, urllib.error, sys

API_URL = "http://localhost:18082/v1/chat/completions"
MODEL = "qwen3.6-128k"

def call_api(messages, max_tokens, timeout=600):
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

def run_decode_benchmark(prompt_text, decode_tokens_list, repeats=3):
    messages = [{"role": "user", "content": prompt_text}]

    # Step 1: 测 prefill baseline（decode=1）
    print(f"\n[Prefill baseline] prompt len 测量...", flush=True)
    prefill_times = []
    for i in range(repeats):
        r, elapsed = call_api(messages, max_tokens=1)
        p_tokens = r["usage"]["prompt_tokens"]
        prefill_times.append(elapsed * 1000)
        print(f"  run{i+1}: {elapsed*1000:.0f}ms  p={p_tokens}", flush=True)
    prefill_ms = statistics.median(prefill_times)
    print(f"  Prefill baseline (median): {prefill_ms:.0f}ms  p_tokens={p_tokens}", flush=True)

    # Step 2: 测不同 decode 长度
    results = []
    for decode_n in decode_tokens_list:
        print(f"\n[Decode={decode_n}]", flush=True)
        e2e_list = []
        actual_decode_list = []
        for i in range(repeats):
            r, elapsed = call_api(messages, max_tokens=decode_n)
            e2e_ms = elapsed * 1000
            actual_decode = r["usage"]["completion_tokens"]
            e2e_list.append(e2e_ms)
            actual_decode_list.append(actual_decode)
            print(f"  run{i+1}: e2e={e2e_ms:.0f}ms  decode={actual_decode}t  finish={r['choices'][0]['finish_reason']}", flush=True)

        e2e_median = statistics.median(e2e_list)
        decode_median = statistics.median(actual_decode_list)
        decode_time_ms = e2e_median - prefill_ms
        if decode_time_ms > 0 and decode_median > 1:
            decode_tps = decode_median / (decode_time_ms / 1000)
            ms_per_token = decode_time_ms / decode_median
        else:
            decode_tps = None
            ms_per_token = None

        result = {
            "decode_target": decode_n,
            "actual_decode_median": decode_median,
            "e2e_median_ms": round(e2e_median, 1),
            "prefill_ms": round(prefill_ms, 1),
            "decode_time_ms": round(decode_time_ms, 1),
            "decode_tps": round(decode_tps, 1) if decode_tps else None,
            "ms_per_token": round(ms_per_token, 1) if ms_per_token else None,
        }
        results.append(result)
        if decode_tps:
            print(f"  decode_time={decode_time_ms:.0f}ms  {decode_tps:.1f} t/s  {ms_per_token:.1f} ms/token", flush=True)

    return prefill_ms, p_tokens, results

def main():
    print("=== Decode 专项性能测试 ===")
    print(f"API: {API_URL}")

    # 等待服务
    for _ in range(12):
        try:
            urllib.request.urlopen("http://localhost:18082/v1/models", timeout=5)
            print("[INFO] 服务就绪\n")
            break
        except:
            time.sleep(5)
    else:
        print("[ERROR] 服务未就绪"); sys.exit(1)

    # 使用短 prompt，避免 prefill 过长影响
    short_prompt = "请写一段关于人工智能发展的文章。"

    prefill_ms, p_tokens, results = run_decode_benchmark(
        prompt_text=short_prompt,
        decode_tokens_list=[50, 100, 200, 400, 800],
        repeats=2
    )

    print("\n\n=== Decode 性能汇总 ===")
    print(f"Prompt: {p_tokens} tokens，Prefill baseline: {prefill_ms:.0f}ms")
    print(f"{'Decode目标':<12} {'实际Decode':<12} {'Decode时间(ms)':<16} {'吞吐(t/s)':<12} {'延迟(ms/t)':<12}")
    print("-" * 65)
    for r in results:
        tps = f"{r['decode_tps']:.1f}" if r['decode_tps'] else "N/A"
        mpt = f"{r['ms_per_token']:.1f}" if r['ms_per_token'] else "N/A"
        print(f"{r['decode_target']:<12} {r['actual_decode_median']:<12} {r['decode_time_ms']:<16.0f} {tps:<12} {mpt:<12}")

    import json
    with open("/tmp/decode_benchmark.json", "w") as f:
        json.dump({"prefill_ms": prefill_ms, "prompt_tokens": p_tokens, "results": results}, f, indent=2)
    print("\n[INFO] 结果写入 /tmp/decode_benchmark.json")

if __name__ == "__main__":
    main()
