# 310P Prefill Performance Optimization Design

## Goal

Improve real-time Prefill behavior for Qwen3.6-35B-A3B-W8A8 on an Atlas 300I
Duo (310P3 x2, TP=2) while preserving the accepted Decode baseline of at least
35 tokens/s.

## Scope and Acceptance Gates

The evaluation environment is fixed to the 128K service configuration and real
model weights. Each candidate is measured with 1K, 4K, 16K, and 32K prompts.

| Metric | Gate |
|---|---|
| 4K and 16K Prefill throughput or TTFT | At least 20% improvement |
| 1K Prefill throughput | No more than 5% regression |
| Decode throughput | At least 35 tokens/s |
| Functional regression | `/v1/models`, text inference, GSM8K, and long-context smoke pass |

All results must include image, commit, CANN/driver versions, service arguments,
prompt token count, warmup count, repetition count, median, and spread.

## Evidence-First Flow

1. Re-run the fixed benchmark matrix and record a new baseline.
2. Collect an operator-level Prefill trace with vLLM `--profiler-config`.
3. Attribute time to scheduling, host-device transfers, attention/mask handling,
   GDN, MoE, GEMM, and collective operations.
4. Select exactly one smallest candidate change from the ranked evidence.
5. Run the same benchmark matrix and all correctness gates before considering a
   second change.

The optimization stops when profiling does not identify a candidate that can
plausibly meet the gates. A result that improves only a microbenchmark is not a
service-level success.

## Candidate Routes

### P0: Recover the Short-Prompt Regression

The latest recorded baseline improves Decode but shows lower peak Prefill around
1K-2K tokens. The first investigation compares the current path with the last
known good configuration and validates the measured delta with operator traces.
This is the preferred route because it addresses TTFT with the lowest semantic
risk.

### P1: Prompt-Length-Aware Chunk Sizing

Existing results show a 4096-token chunk can benefit 3K+ prompts while harming
short prompts. A length-aware choice may be considered only after P0 profiling
proves chunk construction or launch count is material. It must be explicit,
covered by unit tests, deterministic at boundaries, and leave the current safe
path available as the fallback.

### P2: Isolated Custom FlashAttention Evaluation

The `causal_fa_310p` path remains experimental because prior records report a
hang at sequence lengths of 2048 or greater. It is not enabled in the serving
path. Reconsider it only after a standalone NPU kernel test, then real-weight
service inference, long-context stability, and accuracy gates all succeed.

## Safety and Reproducibility

- Do not combine configuration tuning, kernel changes, and scheduler changes in
  one comparison.
- Every run retains raw benchmark JSON and profiler output outside the source
  tree or in a clearly named ignored artifact directory.
- A candidate that causes a crash, output corruption, a Decode regression, or
  an unmet short-prompt gate is rejected and reverted before advancing.
- New environment variables must be centralized in `vllm_ascend/envs.py`; this
  task does not introduce one without review.

## Deliverables

1. A repeatable benchmark and profiler invocation with structured output.
2. A baseline and hotspot report identifying the next implementation target.
3. A focused, test-first optimization change only if the evidence supports it.
4. A final report comparing all acceptance gates and documenting rejected paths.
