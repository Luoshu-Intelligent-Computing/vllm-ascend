#!/usr/bin/env python3
"""
全面性能基准测试：TTFT / 每 token 耗时 / E2E 延时 / 吞吐量 / Decode 速度
使用流式 API 精确测量首 token 时间
"""
import json, time, statistics, urllib.request, urllib.error, sys

API_BASE = "http://localhost:18082"
MODEL = "qwen3.6-128k-nightly"
REPEATS = 3

# ─────────────────────────────────────────
# 流式请求：精确测量 TTFT 和 per-token 时间
# ─────────────────────────────────────────
def call_stream(messages, max_tokens):
    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        f"{API_BASE}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t_start = time.monotonic()
    t_first = None
    token_times = []
    total_tokens = 0
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                now = time.monotonic()
                if t_first is None:
                    t_first = now
                token_times.append(now)
                total_tokens += 1
    t_end = time.monotonic()
    return {
        "ttft_ms":     (t_first - t_start) * 1000 if t_first else None,
        "e2e_ms":      (t_end  - t_start) * 1000,
        "output_tokens": total_tokens,
        "token_times": token_times,
        "t_start":     t_start,
    }

def per_token_stats(r):
    """从 token_times 计算逐 token 间隔"""
    if len(r["token_times"]) < 2:
        return None, None, None
    intervals = [(r["token_times"][i] - r["token_times"][i-1]) * 1000
                 for i in range(1, len(r["token_times"]))]
    return (statistics.mean(intervals),
            statistics.median(intervals),
            max(intervals))

def build_prompt(n_tokens):
    unit = "巴黎是法国的首都，也是欧洲最重要的文化和经济中心之一。"
    repeats = max(1, n_tokens // 12)
    return unit * repeats + "\n\n这段内容讲的是哪个城市？"

def tokenize(text):
    """估算 prompt 实际 token 数"""
    payload = json.dumps({"model": MODEL, "prompt": text}).encode()
    req = urllib.request.Request(f"{API_BASE}/tokenize",
                                  data=payload,
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read()).get("count", -1)
    except:
        return -1

# ─────────────────────────────────────────────
# 测试矩阵
# ─────────────────────────────────────────────
# (label, approx_input_tokens, max_output_tokens)
PERF_CASES = [
    ("in256_out128",    256,   128),
    ("in1k_out128",    1024,   128),
    ("in4k_out128",    4096,   128),
    ("in8k_out128",    8192,   128),
    ("in16k_out128",  16384,   128),
    ("in32k_out128",  32768,   128),
    ("in256_out512",   256,   512),
    ("in1k_out512",   1024,   512),
    ("in4k_out512",   4096,   512),
]

# Decode 专项（极短 prompt，长输出，精确测 decode 速度）
DECODE_CASES = [32, 64, 128, 256, 512]


def run_case(label, prompt_tokens, max_out, reps=REPEATS):
    prompt = build_prompt(prompt_tokens)
    messages = [{"role": "user", "content": prompt}]
    actual_in = tokenize(prompt)

    results = []
    print(f"\n  [{label}] in≈{prompt_tokens} (actual={actual_in}) out={max_out} ×{reps}", flush=True)
    for i in range(reps):
        try:
            r = call_stream(messages, max_out)
            pt_mean, pt_med, pt_max = per_token_stats(r)
            decode_tps = (r["output_tokens"] / (r["e2e_ms"] - r["ttft_ms"]) * 1000
                          if r["ttft_ms"] and r["output_tokens"] > 1
                          else None)
            results.append({
                "ttft_ms":     r["ttft_ms"],
                "e2e_ms":      r["e2e_ms"],
                "output_tokens": r["output_tokens"],
                "pt_mean_ms":  pt_mean,
                "pt_med_ms":   pt_med,
                "decode_tps":  decode_tps,
            })
            print(f"    run{i+1}: TTFT={r['ttft_ms']:.0f}ms E2E={r['e2e_ms']:.0f}ms "
                  f"out={r['output_tokens']}t pt_mean={'N/A' if pt_mean is None else f'{pt_mean:.1f}'}ms "
                  f"decode={'N/A' if decode_tps is None else f'{decode_tps:.1f}'}t/s", flush=True)
        except Exception as e:
            print(f"    run{i+1}: ERROR {e}", flush=True)

    if not results:
        return None
    def med(key): return statistics.median(r[key] for r in results if r[key] is not None)
    return {
        "label": label,
        "actual_input_tokens": actual_in,
        "max_output": max_out,
        "ttft_ms_med":   med("ttft_ms"),
        "e2e_ms_med":    med("e2e_ms"),
        "output_tokens": med("output_tokens"),
        "pt_mean_ms_med": med("pt_mean_ms") if any(r["pt_mean_ms"] for r in results) else None,
        "decode_tps_med": med("decode_tps") if any(r["decode_tps"] for r in results) else None,
        "e2e_tps":       med("output_tokens") / med("e2e_ms") * 1000,
        "prefill_tps":   actual_in / med("ttft_ms") * 1000 if actual_in > 0 else None,
    }

def run_decode_case(decode_n, pf_baseline_ms, reps=REPEATS):
    """极短 prompt + 长输出，用于纯 decode 速度"""
    prompt = "请详细介绍人工智能的发展历史。"
    messages = [{"role": "user", "content": prompt}]

    results = []
    print(f"\n  [decode={decode_n}]", flush=True)
    for i in range(reps):
        try:
            r = call_stream(messages, decode_n)
            dec_ms = r["e2e_ms"] - pf_baseline_ms
            dec_tps = r["output_tokens"] / (dec_ms / 1000) if dec_ms > 0 else None
            mpt = dec_ms / r["output_tokens"] if r["output_tokens"] > 0 else None
            results.append({
                "e2e_ms": r["e2e_ms"],
                "out": r["output_tokens"],
                "dec_ms": dec_ms,
                "dec_tps": dec_tps,
                "mpt": mpt,
            })
            print(f"    run{i+1}: e2e={r['e2e_ms']:.0f}ms out={r['output_tokens']}t "
                  f"dec={dec_ms:.0f}ms {'N/A' if dec_tps is None else f'{dec_tps:.1f}'}t/s "
                  f"{'N/A' if mpt is None else f'{mpt:.1f}'}ms/t", flush=True)
        except Exception as e:
            print(f"    run{i+1}: ERROR {e}", flush=True)

    if not results:
        return None
    def med(k): return statistics.median(r[k] for r in results if r[k] is not None)
    return {
        "decode_target": decode_n,
        "actual_output": med("out"),
        "dec_ms_med":    med("dec_ms"),
        "dec_tps_med":   med("dec_tps"),
        "mpt_med":       med("mpt"),
    }


def main():
    print("=" * 70)
    print("Qwen3.6-35B-A3B-w8a8 @ 310P×2 全面性能测试")
    print(f"服务: {API_BASE}  模型: {MODEL}")
    print(f"参数: max_num_batched_tokens=1024, TP=2, max_model_len=131072")
    print("=" * 70)

    # ── Part 1: Prefill + E2E 综合性能 ──────────────────────
    print("\n\n=== Part 1: Prefill / E2E 性能（不同输入/输出规格）===")
    perf_results = []
    for label, in_tok, out_tok in PERF_CASES:
        res = run_case(label, in_tok, out_tok)
        if res:
            perf_results.append(res)

    # ── Part 2: 纯 Decode 速度 ────────────────────────────
    print("\n\n=== Part 2: 纯 Decode 速度（首先测 Prefill baseline）===")
    # 用极短 prompt 跑 decode=1 获取 prefill baseline
    short_prompt = "请详细介绍人工智能的发展历史。"
    pf_times = []
    for _ in range(3):
        r = call_stream([{"role": "user", "content": short_prompt}], 1)
        pf_times.append(r["e2e_ms"])
        print(f"  prefill-baseline run: {r['e2e_ms']:.0f}ms", flush=True)
    pf_baseline = statistics.median(pf_times)
    print(f"  Prefill baseline (median, decode=1): {pf_baseline:.0f}ms\n")

    decode_results = []
    for d in DECODE_CASES:
        res = run_decode_case(d, pf_baseline)
        if res:
            decode_results.append(res)

    # ── 打印汇总 ──────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("综合性能汇总")
    print("=" * 70)
    print(f"\n{'标签':<20} {'实际输入':>8} {'max_out':>7} {'TTFT(ms)':>10} {'E2E(ms)':>9} {'实际输出':>8} {'Prefill(t/s)':>13} {'decode(t/s)':>12} {'ms/token':>9}")
    print("-" * 110)
    for r in perf_results:
        dtps = 'N/A' if r['decode_tps_med'] is None else f"{r['decode_tps_med']:.1f}"
        mpt  = 'N/A' if r['pt_mean_ms_med'] is None else f"{r['pt_mean_ms_med']:.1f}"
        print(f"{r['label']:<20} {r['actual_input_tokens']:>8} {r['max_output']:>7} "
              f"{r['ttft_ms_med']:>10.0f} {r['e2e_ms_med']:>9.0f} {r['output_tokens']:>8.0f} "
              f"{r['prefill_tps']:>13.1f} {dtps:>12} {mpt:>9}")

    print(f"\n\n{'decode目标':>12} {'实际输出':>8} {'decode时间ms':>14} {'decode t/s':>12} {'ms/token':>10}")
    print("-" * 60)
    for r in decode_results:
        print(f"{r['decode_target']:>12} {r['actual_output']:>8.0f} {r['dec_ms_med']:>14.0f} "
              f"{r['dec_tps_med']:>12.1f} {r['mpt_med']:>10.1f}")

    # 保存 JSON
    out = {
        "model": MODEL,
        "config": "max_num_batched_tokens=1024, TP=2, max_model_len=131072",
        "prefill_baseline_ms": pf_baseline,
        "perf_cases": perf_results,
        "decode_cases": decode_results,
    }
    with open("/tmp/perf_detailed.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n\n详细数据已保存: /tmp/perf_detailed.json")


if __name__ == "__main__":
    main()
