# PI-1 Live SplitFuse Mask Width Design

## Goal

Remove the fixed 128K SplitFuse mask width from the chunked-prefill hot path
without changing causal semantics or the FRACTAL_NZ layout expected by the
legacy 310P operator.

## Decision

`AttentionMaskBuilder310.get_splitfuse_mask()` will use the maximum value in
the batch's `seq_lens` as its K width. It will retain the existing positions,
the `col_idx > positions` causal rule, `nd_to_nz_spec()` padding, and the
FRACTAL_NZ format cast. No environment variable or scheduling policy changes.

The live width is safe for the validated operator contract: a dedicated ATB
probe accepted non-16-aligned K=33 (physical NZ K=48) and produced the same
output as K=64 with the inactive tail set to `-inf`.

## Alternatives

- Keep the fixed configured width: no behavioral risk, but preserves the
  measured `Greater` hotspot.
- Reuse a fixed-sized mask cache: reduces allocation churn but does not remove
  the broadcast work or its 128K K dimension.
- Use a compressed-mask operator: not available in the active CANN runtime.

## Unit Validation

Focused CPU-mocked unit tests will require K to track the live maximum,
including a non-16-aligned width, and will reconstruct the ND mask to verify
per-token causal values. The production change then requires the frozen
AISBench matrix, GSM8K quick sample, long-context smoke test, Decode checks,
and a bounded profiler confirmation before acceptance.

The implementation completed the RED/GREEN cycle in the remote v11 image: the
first test failed with the fixed `[1,256,16,16]` mask for live K=7, then the
complete target module passed three tests after the minimal source change. The
K=33 test reconstructs physical `[1,3,16,16]` NZ layout and verifies its
zero-padded tail. This is unit evidence only; service validation remains open.

## Rollback

The change is one source file and will be deployed in a new image/container
name. Stopping that container and starting the preserved
`vllm-prefill-baseline-v11` container restores the current baseline.
