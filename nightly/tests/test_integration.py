#!/usr/bin/env python3
"""
test_integration.py - 测试 causal_fa_310p 集成到 vllm-ascend 310P 路径

验证项：
1. causal_fa_310p 算子可加载
2. API 调用正确（query/key/value/seq_lens/scale）
3. 输出无 NaN/Inf，形状正确
"""
import torch
import torch_npu
import os
import sys

def test_operator_loading():
    """测试算子加载"""
    print("=" * 60)
    print("Test 1: Operator Loading")
    print("=" * 60)

    so_path = "/opt/causal_fa_310p/torch_npu_causal_fa.cpython-311-aarch64-linux-gnu.so"

    if not os.path.exists(so_path):
        print(f"❌ ERROR: Operator .so not found at {so_path}")
        print("   Please ensure volume mount: -v patches/310p-long-context/causal_fa_ops:/opt/causal_fa_310p:ro")
        return False

    try:
        torch.ops.load_library(so_path)
        print(f"✅ SUCCESS: Loaded {so_path}")
    except Exception as e:
        print(f"❌ ERROR: Failed to load operator: {e}")
        return False

    # Check if operator is registered
    if hasattr(torch.ops, 'npu_ext') and hasattr(torch.ops.npu_ext, 'causal_fa_310p'):
        print("✅ SUCCESS: torch.ops.npu_ext.causal_fa_310p is available")
        return True
    else:
        print("❌ ERROR: torch.ops.npu_ext.causal_fa_310p not registered")
        return False

def test_operator_call():
    """测试算子调用"""
    print("\n" + "=" * 60)
    print("Test 2: Operator Call")
    print("=" * 60)

    try:
        torch_npu.npu.set_device(0)

        # Realistic parameters (similar to Qwen3.5-35B-MoE attention)
        T = 128  # Shorter for quick test
        N = 28   # num_heads (Qwen3.5-35B-MoE)
        N_kv = 4 # num_kv_heads (GQA G=7)
        D = 128  # head_dim
        B = 2    # batch size
        scale = 1.0 / (D ** 0.5)

        print(f"Parameters: T={T}, N={N}, N_kv={N_kv}, D={D}, B={B}, scale={scale:.6f}")

        # Create test data
        torch.manual_seed(42)
        query = torch.randn(T, N, D, dtype=torch.float16, device="npu:0")
        key = torch.randn(T, N_kv, D, dtype=torch.float16, device="npu:0")
        value = torch.randn(T, N_kv, D, dtype=torch.float16, device="npu:0")

        # seq_lens: variable length batch
        seq_lens = torch.tensor([64, 64], dtype=torch.int32, device="cpu")

        print(f"Input shapes: query={query.shape}, key={key.shape}, value={value.shape}")
        print(f"seq_lens: {seq_lens.tolist()}")

        # Call operator
        output = torch.ops.npu_ext.causal_fa_310p(query, key, value, seq_lens, scale)

        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
        print(f"Output device: {output.device}")

        # Sanity checks
        if output.shape != query.shape:
            print(f"❌ ERROR: Output shape mismatch (expected {query.shape}, got {output.shape})")
            return False

        if output.isnan().any():
            print("❌ ERROR: Output contains NaN")
            return False

        if output.isinf().any():
            print("❌ ERROR: Output contains Inf")
            return False

        if output.abs().max() < 1e-6:
            print("❌ ERROR: Output is all zeros")
            return False

        print(f"Output stats: mean={output.mean().item():.4f}, std={output.std().item():.4f}")
        print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        print("✅ SUCCESS: Operator call completed, output looks reasonable")
        return True

    except Exception as e:
        print(f"❌ ERROR: Operator call failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dtype_conversion():
    """测试 dtype 转换（seq_lens 必须是 int32）"""
    print("\n" + "=" * 60)
    print("Test 3: Dtype Conversion")
    print("=" * 60)

    try:
        torch_npu.npu.set_device(0)

        T, N, N_kv, D = 16, 8, 4, 128
        scale = 1.0 / (D ** 0.5)

        query = torch.randn(T, N, D, dtype=torch.float16, device="npu:0")
        key = torch.randn(T, N_kv, D, dtype=torch.float16, device="npu:0")
        value = torch.randn(T, N_kv, D, dtype=torch.float16, device="npu:0")

        # Test with int64 seq_lens (should work after conversion)
        seq_lens_int64 = torch.tensor([T], dtype=torch.int64, device="cpu")
        seq_lens_int32 = seq_lens_int64.to(torch.int32)

        print(f"seq_lens dtype before: {seq_lens_int64.dtype}")
        print(f"seq_lens dtype after: {seq_lens_int32.dtype}")

        output = torch.ops.npu_ext.causal_fa_310p(query, key, value, seq_lens_int32, scale)

        print("✅ SUCCESS: Dtype conversion handled correctly")
        return True

    except Exception as e:
        print(f"❌ ERROR: Dtype conversion test failed: {e}")
        return False

def main():
    print("Testing causal_fa_310p integration with vllm-ascend 310P")
    print()

    results = []

    # Run tests
    results.append(("Operator Loading", test_operator_loading()))
    if results[-1][1]:  # Only proceed if loading succeeded
        results.append(("Operator Call", test_operator_call()))
        results.append(("Dtype Conversion", test_dtype_conversion()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    all_passed = all(result[1] for result in results)
    print("=" * 60)
    if all_passed:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
