# PI-1 Live SplitFuse Mask Width Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Generate legacy 310P SplitFuse masks only as wide as the active
batch's maximum context length.

**Architecture:** Keep the existing causal broadcast and FRACTAL_NZ conversion.
Replace only the broadcast's configured K source with the current maximum
sequence length, which is already copied to CPU by the method for position
construction.

**Tech Stack:** Python, PyTorch, torch_npu mocked unit tests, pytest.

---

### Task 1: Add a failing live-width shape test

**Files:**
- Modify: `tests/ut/_310p/attention/test_attention_mask_310.py`

**Step 1: Write the failing test**

Add `test_get_splitfuse_attn_mask_uses_live_max_context` with
`query_start_loc=[0, 1, 5]` and `seq_lens=[7, 4]`. Patch
`torch_npu.npu_format_cast` to return its tensor input and assert the returned
NZ-layout shape is `(1, 1, 16, 16)`, not the builder's 4096-wide shape.

**Step 2: Run the test to verify it fails**

Run in the isolated `ails-a5` v11 test container:

```bash
pytest -q tests/ut/_310p/attention/test_attention_mask_310.py::TestAttentionMaskBuilder310::test_get_splitfuse_attn_mask_uses_live_max_context
```

Expected: failure because the current implementation creates
`(1, 256, 16, 16)` from `cls.max_seqlen=4096`.

### Task 2: Implement the minimal live-width change

**Files:**
- Modify: `vllm_ascend/_310p/attention/attention_mask.py:95-110`

**Step 1: Replace the fixed width**

After building `c_list`, derive `max_context_len = max(c_list)`. Replace:

```python
col_idx = torch.arange(cls.max_seqlen, dtype=torch.int64, device=device)
```

with:

```python
col_idx = torch.arange(max_context_len, dtype=torch.int64, device=device)
```

Do not alter the position rule, dtype, padding helper, or format cast.

**Step 2: Run the targeted test to verify it passes**

Run the command from Task 1. Expected: PASS.

Result: the targeted test passed in the remote v11 container after the minimal
source change.

### Task 3: Add and verify causal/non-aligned coverage

**Files:**
- Modify: `tests/ut/_310p/attention/test_attention_mask_310.py`

**Step 1: Write the failing test**

Add `test_get_splitfuse_attn_mask_preserves_causality_for_non_aligned_width`.
Use `query_start_loc=[0, 1, 3]` and `seq_lens=[33, 5]`. Reverse the mocked NZ
layout with `permute(0, 2, 1, 3).reshape(16, 48)`, then assert the first three
rows at valid K=33 contain exactly the causal values for positions 32, 3, and
4. Assert the non-aligned physical K tail (columns 33--47) is zero padded.

**Step 2: Run it to verify the current implementation passes only after Task 2**

```bash
pytest -q tests/ut/_310p/attention/test_attention_mask_310.py::TestAttentionMaskBuilder310::test_get_splitfuse_attn_mask_preserves_causality_for_non_aligned_width
```

Expected: PASS after Task 2. Its failure before Task 2 is a shape mismatch,
which confirms it detects the fixed-width behavior.

**Step 3: Run the full target module**

```bash
pytest -q tests/ut/_310p/attention/test_attention_mask_310.py
```

Expected: all tests pass.

Result: `3 passed, 14 warnings in 0.07s` in the remote one-NPU test container.
The live-K ATB ABI was separately proven before this implementation.

### Task 4: Review and prepare service validation

**Files:**
- Modify: `.agent/dev-docs/310p-prefill-performance/LOG.md`
- Modify: `.agent/dev-docs/310p-prefill-performance/cannbot/progress.md`

**Step 1: Run static review**

```bash
ruff check vllm_ascend/_310p/attention/attention_mask.py tests/ut/_310p/attention/test_attention_mask_310.py
```

**Step 2: Record the unit-test evidence**

Record test command, source commit, and test outcome before building a distinct
PI-1 image. Do not overwrite the baseline image or container.

**Step 3: Commit**

```bash
git add vllm_ascend/_310p/attention/attention_mask.py tests/ut/_310p/attention/test_attention_mask_310.py docs/plans/
git commit -s -m "perf(310p): narrow splitfuse mask to live context"
```

Status: awaiting explicit confirmation for the signed commit and the subsequent
distinct-image build. The preserved baseline image and container are not targets
of either action.
