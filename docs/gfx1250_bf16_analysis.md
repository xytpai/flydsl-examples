# gfx1250 BF16 8192-cubed: ISA, ATT and power-state analysis

## Result

The existing no-spill kernel at `5477ef0` already exceeds 2.5 PFLOPS in
short bursts. The much lower continuous-load result is primarily a power-state
effect shared by the C++ PeakGemm baseline, not a BF16 accuracy or dispatch bug.

f01, physical GPUs 1 and 2 tested separately, 2026-11-02. Both kernels use
BF16 input/output, FP32 accumulation, NT, tile 256x256x128, 4x4 wave32,
2 stages, group_m=0, no bias or split-K. All mathematical outputs are checked;
neither precision nor workload was reduced.

| Protocol | FlyDSL TFLOPS (6 run medians) | PeakGemm C++ TFLOPS (6 run medians) |
|---|---:|---:|
| Burst | 2838.4–2866.6 | 2853.0–2897.5 |
| Sustained | 2211.8–2237.5 | 2249.1–2262.3 |

These are ranges of **run medians**, not fastest individual samples.
The kernel was unchanged during this study. No power limit, clock, system
setting or other user's workload was modified.

### Fixed measurement protocols

- Same A/B/C pointers for the two implementations; random signed binary-fraction
  inputs. Full-output equality checked against independent FP32 matmul and after
  changing A in captured graphs. GPU process monitoring rejected contaminated
  runs.
- Eight GEMM launches per HIP graph. Three repetitions on each GPU, alternating
  implementation order between repetitions.
- **Burst:** synchronize, idle 0.5 s, replay one warmup graph, then time all five
  graph replays separately and take their median.
- **Sustained:** continuously replay with completion synchronization for at least
  0.5 s, then take the median of 21 samples, each containing four graph replays.
- Allocation, reference calculation, compilation and capture are outside timing.
  The idle interval is explicit: burst performance is **not** sustained throughput.

## Power-state evidence

A separate 200-sample continuous series was correlated with read-only AMD SMI
telemetry at approximately 20 ms intervals. Representative observations:

| Series index | GEMM us | GFX MHz | Socket W | Hotspot C |
|---|---:|---:|---:|---:|
| 0 | 418.7 | 2355 | 1160 | 49 |
| 10 | 385.5 | 2191 | 1817 | 49 |
| 30 | 459.7 | 1941 | 2275 | 49 |
| 70 | 470.7 | 1593 | 2433 | 50 |
| 170 | 472.8 | 1513 | 2500 | 52 |

Telemetry samples are nearest in host time, not per-instruction clock readings.
The power limit reported by SMI is 2500 W; `ppt_residency_acc` increases during
the slowdown. Temperature stays near 49–52 C. GFX returns to about 2373 MHz
after the load ends. Idle before/after snapshots alone missed the in-load
frequency drop.

The clock/power transition and the same burst-to-sustained drop in C++ strongly
support PPT power limiting as the main reason the long test does not sustain
2.5 PFLOPS. This does not establish the hardware's maximum achievable efficiency
or promise that a different kernel could never sustain that target.

## Assembly and Advanced Thread Trace

ROCm rocprofv3 1.3.5, ATT decoder 0.2.2. Trace targets one warmed BF16 GEMM,
shader-engine mask 1, selected SIMD 0; separate captures on CU/WGP 1 and 2.
Both traces decode 16 full waves, 4096 WMMA instructions per wave, **65536 WMMA
total**, with zero unmapped instruction PCs. Code objects and raw ATT are retained.

The pair main loop executes, for both implementations:

- 128 `v_wmma_f32_16x16x32_bf16` instructions;
- 128 `ds_load_b128` instructions;
- 32 `s_wait_dscnt` instructions;
- four `tensor_load_to_lds` instructions and two workgroup barriers.

Both have zero scratch. Current FlyDSL uses 212 VGPRs; the C++ reference uses 222.
Profiler register fields use a different encoding: the code-object metadata is
the resource-count reference.

### Sampled-wave stall totals, normalized per 1024 WMMA

| Capture | Implementation | LDS wait | Barrier wait | TDM wait | WMMA issue stall |
|---|---|---:|---:|---:|---:|
| GPU0 CU1 | FlyDSL | 8485 | 6484 | 564 | 2509 |
| GPU0 CU1 | C++ | 8003 | 4178 | 494 | 2381 |
| GPU0 CU2 | FlyDSL | 8599 | 5682 | 445 | 2589 |
| GPU0 CU2 | C++ | 7889 | 4724 | 436 | 2422 |

In the CU1 FlyDSL capture, main-loop barrier PCs `0x1d70` and `0x21fc` average
446.5 and 312.2 stall cycles per hit. The first LDS waits at `0x1df0` and
`0x229c` average 138.6 and 149.7 cycles per hit. The main TDM wait at `0x1d20`
averages only 47.9. These PCs belong to the retained code object, not universal
addresses across compiler revisions.

The remaining kernel-local gap points to LDS-to-WMMA scheduling and differences
in wave arrival at barriers, not BF16 conversion as the dominant cost.
**Wave stalls are not GPU wall time and cannot be summed into a TFLOPS estimate.**
ATT changes queue interception and timing; all performance above is measured
without profiler. Long-prewarm captures also remain complete and show the same
general instruction mix.

## Single-variable experiments

All variants below are independent of the delivered kernel.

| Experiment | Correctness | Finding |
|---|---|---|
| Disable redundant TDM boundary updates under the existing full-tile contract | 105 tests pass | Fewer scalar instructions, no meaningful performance gain |
| Prefetch all A/B fragments for the current K32 step | 105 tests pass | 244 VGPR, zero spill; about 0.4–1.5% gain in tested continuous runs |
| Native vector BF16 pair conversion | 105 tests pass | Conversion instructions 128→64, but throughput unchanged |
| Remove B-fragment compiler scheduling barrier | 105 tests pass | About 1% gain; small and workload dependent |
| Remove A-fragment compiler scheduling barrier | 105 tests pass | Approximately 7% slower |
| Opaque inline-asm BF16 pair conversion | **Fails** | Rejected; adding 8/32-cycle diagnostic NOPs does not fix all cases |

For prefetching, a paired GPU1 ATT capture reduces LDS-wait stall from 8475 to
4640 per1024 WMMA, but increases barrier and WMMA issue stall substantially.
It mostly moves the waiting rather than eliminating it; the wall-time gain is
small. No hardware synchronization barrier was removed.

The standalone conversion instruction probe confirms low/high BF16 packing.
The opaque-asm failure's full compiler/hazard cause is **not established**;
it must not be called a solved hardware conversion bug. Native conversion is
correct but not a useful optimization here.

A 64-configuration discovery scan and a 15-configuration sustained screen were
also performed. The original default tile remains the best or near-best within
that bounded set. No global-optimum claim. An initial GPU0 discovery run was
contaminated by a foreign process and discarded; the GPU1 discovery rerun was
clean. Short discovery timings are not used as sustained results.

## Reproduce with the repository benchmark

On an idle gfx1250, with the validated FlyDSL 0.4.1 setup:

    python test_gemm_a16w16_tdm_gfx1250.py --mnk 8192 8192 8192 --dtype bf16 --mode both --json bf16_phases.json

The output labels every burst/sustained repetition and the JSON saves every
sample. The default `--mode legacy` retains the original benchmark's 20-node,
9-sample protocol; it does not silently switch to a burst or best-of metric.

The final repository CLI was also run directly on idle GPU1 under process
monitoring. `--mode legacy` measured **2700.65 TFLOPS** (0.407129 ms), preserving
the original 20-node/9-sample timing rather than using a burst override.
`--mode both` produced burst medians 2845.47/2835.84/2847.98 TFLOPS and
sustained medians 2317.97/2244.71/2228.58 TFLOPS. These are separate runs from
the controlled two-implementation table above and are not a performance gain
caused by the benchmark update. The full suite passed **105 tests**.

[Raw timing samples and power correlation](gfx1250_bf16_measurements.json)
include the fixed-protocol repeats and the repository CLI runs.

No experimental kernel change was promoted by this investigation. The delivered
changes are explicit benchmark modes and this analysis.
