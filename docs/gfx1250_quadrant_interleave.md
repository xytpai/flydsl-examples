# gfx1250 quadrant interleaving

The existing `build_gemm_gfx1250` kernel now accepts `interleave=False`.
Both public wrappers accept the same optional keyword:

    gemm_a16w16_gfx1250(
        a, b, block_m=256, block_n=256, block_k=128,
        stages=2, m_waves=2, n_waves=2, interleave=True,
    )

`a` is logical `[M,K]`; `b` is a logical `[K,N]` NT view. FP16, BF16 and
FP8 PTPC are supported. FP8 keeps its existing per-row/per-column scales.

## Schedule

A wave's accumulator tile is split into four equal quadrants. The two
traversal orders are:

- even waves: Q00 -> Q01 -> Q11 -> Q10;
- odd waves: Q00 -> Q10 -> Q11 -> Q01.

The first two quadrants interleave independent LDS loads of the next half
of A or B with WMMA. The last two quadrants preload the next K step's first
halves only after their old registers are no longer read. This reduces
simultaneous fragment lifetimes and staggers A/B LDS reads between waves;
it does not assume fixed wave-to-SIMD placement.

The hot loop unrolls two K tiles. Stage indices are literal 0/1 for the
two-buffer case; deeper rings, odd trip counts and the final drain remain
explicit. At tile boundaries all waves finish reading the old LDS slot
and wait for the next TDM tile before the slot is reused. Unused prefetches
are clamped to the split's last tile, including the existing stages-1
minimum-length case.

There is still only one GPU kernel definition. Both schedules reuse the
file's `tdm_operand` and `copy_operand` helpers; no producer-wave roles are
introduced. The optional schedule uses swapped WMMA operands for
row-contiguous accumulator fragments, restores logical C in a 16-byte-padded
C-shuffle, then uses the same TDM operand helper for no-bias, unsplit output
stores. Bias and split-K retain the existing output arithmetic.

`interleave=True` requires NT and even wave tiles with enough WMMA slots
to cover the independent fragment loads. Invalid combinations raise
`ValueError`; no silent schedule fallback. The flag is included in the
compile factory, constexpr cache parameter and kernel name. Default kernel
names and the default schedule remain unchanged.

This change does **not** introduce AITER's atomic/workspace policy or tuner,
nor change the existing split-K rounding, FP8 scaling or block-swizzle rules.
In particular, this branch still rounds split partials before its final
FP32 sum; equivalence with that existing implementation is not a claim
that split-K equals an unsplit FP32 GEMM.

## Tests

    python -m pytest -q test_gemm_gfx1250_interleave.py

On C5 gfx1250 / FlyDSL 0.5.0:
- 98 new cases passed: BF16/FP16/FP8, shortest stages-1 prologues,
  odd/even tile counts, stages2/3/4, splits1/2/3/5/9, rectangular
  wave tiles, optional bias, offsets, guarded output, graph input/scale
  mutation, odd N, explicit CPU FP32 reference and configuration rejection.
- 38 selected existing tests passed, covering default allocation, strides,
  offsets, invalid inputs, swizzle edges, wide K, streams and FP8 all-paths.
  The remaining 577 cases in that file were deselected, not claimed passed.

## Same-configuration performance

8192 cubed, tile256x256, stages2, group_m0, split1, NT, no bias.
BK128 for BF16/FP16 and BK256 for FP8. Physical C5 GPU3, FlyDSL0.5.0.
Two fresh processes, two independently captured graphs per schedule,
four calls per graph and 31 randomized timing rounds. Table values are
means of the four per-graph medians. Every timed output was bit-identical
between schedules on the tested inputs.

| dtype | waves | original us | interleaved us | interleaved TFLOPS |
|---|---|---:|---:|---:|
| BF16 | 2x2 | 644.90 | 407.40 | 2699 |
| BF16 | 2x4 | 529.45 | 450.18 | 2442 |
| FP16 | 2x2 | 648.52 | 436.04 | 2522 |
| FP16 | 2x4 | 556.46 | 473.71 | 2321 |
| FP8 PTPC | 2x2 | 235.47 | 180.90 | 6078 |

Other, separate tuning jobs were active on GPUs0–2; GPU3 had no competing
GPU job. Clocks were not locked. Original-path A/A differences reached
roughly 16%, and FP8 original-path process timings varied; the table is
local evidence, not a statistically guaranteed speedup or universal peak.
No full-shape retune or LLM serving benefit is claimed.
