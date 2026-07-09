#!/usr/bin/env python3
"""
128K 精度边界测试脚本
目标：定位精度下降的临界点（从 32k 到 128k 逐步测试）
"""
import json, urllib.request, urllib.error, sys

API_URL = "http://localhost:18082/v1/chat/completions"
MODEL = "qwen3.6-128k-nightly"

def call_api(messages, max_tokens=100, temperature=0.0, timeout=600):
    """调用 API"""
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
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read()), None
    except Exception as e:
        return None, str(e)

def build_long_prompt(target_tokens):
    """构建指定长度的 prompt"""
    # 基础单元约 12 tokens
    base = "巴黎是法国的首都，也是欧洲最重要的文化和经济中心之一。"
    repeats = max(1, target_tokens // 12)
    content = base * repeats
    question = "\n\n请问本文中提到的城市叫什么名字？请直接回答城市名称。"
    return content + question

def test_context_length(target_tokens, label):
    """测试指定上下文长度"""
    print(f"\n{'='*60}")
    print(f"测试：{label}（目标 ~{target_tokens} tokens）")
    print(f"{'='*60}")

    prompt = build_long_prompt(target_tokens)
    messages = [
        {"role": "system", "content": "你是一个简洁的助手，直接回答问题。"},
        {"role": "user", "content": prompt}
    ]

    result, error = call_api(messages, max_tokens=100, temperature=0.0)

    if error:
        print(f"❌ 错误: {error}")
        return {
            "label": label,
            "target_tokens": target_tokens,
            "status": "error",
            "error": error
        }

    usage = result.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    content = result["choices"][0]["message"].get("content", "")
    finish_reason = result["choices"][0]["finish_reason"]

    # 检查输出质量
    has_paris = "巴黎" in content
    is_gibberish = any([
        len(content) > 200,  # 过长（应该简短回答）
        content.count(content[:10]) > 3 if len(content) > 10 else False,  # 重复
        len(set(content)) < len(content) * 0.3 if len(content) > 20 else False,  # 字符重复率高
    ])

    status = "✅ 正常" if has_paris and not is_gibberish else "❌ 异常"

    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Completion tokens: {completion_tokens}")
    print(f"Finish reason: {finish_reason}")
    print(f"输出内容: {content[:200]}")
    if len(content) > 200:
        print(f"  ... (截断，总长 {len(content)} 字符)")
    print(f"包含'巴黎': {has_paris}")
    print(f"疑似乱码: {is_gibberish}")
    print(f"状态: {status}")

    return {
        "label": label,
        "target_tokens": target_tokens,
        "actual_prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "content": content,
        "has_correct_answer": has_paris,
        "is_gibberish": is_gibberish,
        "status": "normal" if (has_paris and not is_gibberish) else "degraded"
    }

def main():
    print("=" * 60)
    print("128K 精度边界测试")
    print("=" * 60)
    print(f"API: {API_URL}")
    print(f"Model: {MODEL}")
    print()

    # 测试点：从正常区（32k）到异常区（128k）
    test_cases = [
        (32000, "32K（基线）"),
        (48000, "48K"),
        (57000, "57K（已知开始异常）"),
        (60000, "60K"),
        (63000, "63K"),
        (64000, "64K"),
        (65000, "65K"),
        (65536, "65536（64K边界）"),
        (66000, "66K"),
        (70000, "70K"),
        (80000, "80K"),
        (96000, "96K"),
        (128000, "128K"),
    ]

    results = []

    for target, label in test_cases:
        result = test_context_length(target, label)
        results.append(result)

        # 如果连续 3 个正常或连续 3 个异常，可以跳过部分测试加速
        if len(results) >= 3:
            last_3 = [r["status"] for r in results[-3:]]
            if all(s == "normal" for s in last_3):
                print("\n⏩ 连续 3 次正常，跳过中间测试点，直接测试更高值")
            elif all(s == "degraded" for s in last_3):
                print("\n⏩ 连续 3 次异常，已定位异常区间，停止测试")
                break

    # 汇总结果
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    print(f"{'标签':<20} {'Prompt Tokens':<15} {'状态':<10}")
    print("-" * 60)

    boundary = None
    for i, r in enumerate(results):
        status_symbol = "✅" if r["status"] == "normal" else "❌"
        print(f"{r['label']:<20} {r['actual_prompt_tokens']:<15} {status_symbol} {r['status']}")

        # 寻找边界（正常→异常）
        if i > 0 and results[i-1]["status"] == "normal" and r["status"] == "degraded":
            boundary = (results[i-1]["actual_prompt_tokens"], r["actual_prompt_tokens"])

    if boundary:
        print(f"\n💡 精度下降边界：{boundary[0]} - {boundary[1]} tokens 之间")
    else:
        print("\n💡 未找到明确边界（全部正常或全部异常）")

    # 保存结果
    output_path = "/tmp/128k_boundary_test.json"
    with open(output_path, "w") as f:
        json.dump({
            "test_cases": results,
            "boundary": boundary,
            "summary": {
                "total": len(results),
                "normal": sum(1 for r in results if r["status"] == "normal"),
                "degraded": sum(1 for r in results if r["status"] == "degraded"),
            }
        }, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存: {output_path}")

if __name__ == "__main__":
    main()
