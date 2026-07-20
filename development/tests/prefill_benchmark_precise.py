#!/usr/bin/env python3
"""
Prefill 精确性能测试（差值法）
方法：E2E(decode=N) - E2E(decode=1) ≈ decode 时间
     Prefill 时间 = E2E(decode=1) - 网络延迟（忽略不计）
     Prefill 吞吐 = prompt_tokens / prefill_time
"""
import json, time, statistics, urllib.request, urllib.error, sys

API_URL = "http://localhost:18082/v1/chat/completions"
MODEL = "qwen3.6-128k"
DECODE_MS_PER_TOKEN = 29.2  # 从 decode_benchmark 测得，ms/token

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
    return r, (time.monotonic() - t0) * 1000

def build_prompt(n_tokens):
    unit = "巴黎是法国的首都，也是欧洲最重要的城市之一。"  # ~12 tokens
    return unit * (n_tokens // 12) + "\n\n请问本文提到的城市是哪里？"

def main():
    print("=== Prefill 精确性能测试（差值法）===")
    print(f"Model: {MODEL}")
    print(f"方法：prefill_time = E2E(decode=1)，prefill_tps = prompt_tokens / prefill_time")
    print(f"decode_ms_per_token = {DECODE_MS_PER_TOKEN} ms（来自 decode_benchmark）\n")

    # 等待服务就绪
    for _ in range(12):
        try:
            urllib.request.urlopen(f"http://localhost:18082/v1/models", timeout=5)
            print("[INFO] 服务就绪\n")
            break
        except:
            time.sleep(5)
    else:
        print("[ERROR] 服务未就绪"); sys.exit(1)

    targets = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    results = []

    for target in targets:
        prompt = build_prompt(target)
        msgs = [{"role": "user", "content": prompt}]
        times = []
        p_tok = 0
        print(f"[{target}t]", end="", flush=True)
        for i in range(3):  # 3次取中位数，更稳定
            try:
                r, ms = call_api(msgs, max_tokens=1)
                p_tok = r["usage"]["prompt_tokens"]
                c_tok = r["usage"]["completion_tokens"]
                # 减去实际 decode 时间（c_tok * decode_ms_per_token）
                prefill_ms = ms - c_tok * DECODE_MS_PER_TOKEN
                times.append(prefill_ms)
                print(f" {ms:.0f}ms(p{p_tok}c{c_tok})", end="", flush=True)
            except Exception as e:
                print(f" ERR:{e}", end="", flush=True)
        print()

        if times:
            med_prefill_ms = statistics.median(times)
            tps = p_tok / (med_prefill_ms / 1000) if med_prefill_ms > 0 else 0
            results.append((target, p_tok, med_prefill_ms, tps))

    # 旧基线（2026-06-29，max_num_batched_tokens=1024，310p-opt-20260709）
    old_baseline = {
        256: 230, 512: None, 1024: 770, 2048: 580,
        4096: 780, 8192: 633, 16384: 960, 32768: 890
    }

    print("\n=== Prefill 性能汇总（差值法）===")
    print(f"{'规模':>8} {'实际tokens':>10} {'Prefill(ms)':>12} {'吞吐(t/s)':>12} {'旧基线':>10} {'变化':>8}")
    print("-" * 65)
    for target, p, ms, tps in results:
        old = old_baseline.get(target)
        if old:
            change = f"{(tps/old - 1)*100:+.1f}%"
        else:
            change = "N/A"
        old_str = str(old) if old else "N/A"
        print(f"{target:>8} {p:>10} {ms:>12.0f} {tps:>12.0f} {old_str:>10} {change:>8}")

    import json as js
    with open("/tmp/prefill_benchmark_precise.json", "w") as f:
        js.dump([{"target": t, "prompt_tokens": p, "prefill_ms": round(ms,1), "tps": round(tps,1)}
                 for t, p, ms, tps in results], f, indent=2)
    print("\n[INFO] 结果写入 /tmp/prefill_benchmark_precise.json")

if __name__ == "__main__":
    main()
