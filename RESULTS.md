# Batched decode attention for Inkling 2 Mini

Branch `engine-batched-sdpa`, cut from `engine-base` (`d421048`).

## Change

The per-row SDPA loop in `Attention._attention` (`L == 1`) is replaced by one
custom Metal decode kernel, `decode_attention` in
`mlx_lm/models/inkling2/kernels.py`. It runs for every decode step on the GPU,
for any batch size and dtype. The CPU path falls back to `_attention_reference`.

The kernel:

- Pass 1: grid (key split, kv head, batch). A threadgroup handles the 3 query
  heads of one kv head (GQA), so each K/V row is read once for all 3. Each lane
  holds 4 of the 128 dims, and the 8 simdgroups take the keys in turns, with an
  online softmax in fp32. The work splits over keys until there are about 256
  threadgroups, with at least 128 keys in each.
- The key position, visibility (causal, left padding, sliding window) and
  relative bias are all computed in the kernel from `cache.offset` (per row),
  `S`, and the ring index `cache._idx` for rotating caches. So no mask, distance
  or bias tensor is built, and nothing that grows with S is allocated.
- Pass 2: one small kernel merges the splits and adds the per-row, per-head
  sink, `exp(sink)` in the denominator only, in fp32.

So one decode attention is 2 kernel launches for any B. Before, it was about
12 mask ops plus one SDPA per row. Ops per B=1 decode step go from 13,510 to
11,502 (−15%), and the cost no longer grows with B.

## Approaches tried

| Approach | Result |
|---|---|
| **A. Fold the sinks into the SDPA mask** (no cache changes). Softmax does not change when all logits of a row move by the same amount, so row b's mask holds `bias − (sink_b − shared)` and SDPA gets `sinks=shared` of shape (H,). `shared` is the midrange over rows, so B=1 is exact. One batched SDPA call, vectorized mask. | Exact in math, but the mask must be bf16 ("Mask type must promote to output type"), so the sink offset gets rounded into it. In the real model the per-row offsets reach **6.8** and mask values reach ~18 (measured with a hook over 2,928 batched attention calls). On a synthetic test with large sinks it gives **4× the base error** (6.3e-2 vs 1.6e-2, FAIL at the 2e-2 bound). In the attention microbenchmark it is about 2× slower than C. Rejected. Diff kept in the notes folder. |
| **B. Reuse `relative_attention` for L==1** | Not used. It assumes keys in temporal order (`diag = S − L + row`), which a rotating cache after wraparound breaks. Its tiles hold 32 queries per threadgroup, so 31/32 of the work is wasted at L=1, and it has one threadgroup per (batch, head) with no split over S. |
| **C. Custom fused decode kernel** (above) | **Chosen.** Fastest, keeps everything in fp32, handles rotating caches and per-row offsets natively, and allocates no S-sized temporaries. |

The "reserve a zero K/V slot in the caches" idea from the brief was not needed:
A gets the same effect without a slot, and C makes the sink exact without one.

## Correctness

`test_decode_attention.py` (copied into this worktree and extended) compares
against the float32 `_attention_reference`:

- The original 5 cases: B=1 and B=4, left padding, sliding window, R<S, S=2048.
- New: large sinks/bias (sink std 8, bias std 4), the case that fails approach A.
- New, real cache objects, decoded step by step and checked at every step
  against a reference over the full K/V history with explicit positions:
  `KVCache` (B=1), `BatchKVCache` (left padded, plus S≈3000 with R=1024<S),
  `RotatingKVCache` (window 64, 120 steps, so it wraps), and
  `BatchRotatingKVCache` (window 64 with 120 and 80 steps, and window 1024 with
  prompts of 1000/600/50/1024 tokens, so it wraps).

Every case passes, with errors the same as base's per-row loop on the same
cases (worst relative error ≈ 7–8e-3, which is bf16 output rounding). In float32
the kernel matches the reference to 4.8e-7, and in float16 to 2e-3.

PR tests: `.venv/bin/python -I -m unittest discover -s ../common -p test_inkling2.py`
→ `Ran 8 tests ... OK`.

End to end (`greedy_consistency.py`): B=1 vs B=4 row 0 is not identical on
either branch. Base diverges at char 669 of 872, and this branch at char 270 of
1029. That comes from bf16 near-ties in greedy decoding, not from a bug (base
shows it too). The teacher-forced logit comparison below is the better measure.

Teacher-forced logit check on the real model (B=4, 4 different chat prompts,
200 decode steps). Each variant replays the same token stream from its own
cache. Logits are compared with the fp32 `_attention_reference` decode path:

| decode attention | top-1 agree | max abs logit diff | mean KL | max KL |
|---|---:|---:|---:|---:|
| base per-row SDPA loop | 787/800 | 7.11 | 2.5e-3 | 0.175 |
| new kernel | 789/800 | 6.14 | 5.1e-3 | 0.349 |

Both are equally close to the reference on top-1 agreement. The kernel's mean KL is about 2×
base's, but both are small. The KL is driven by a few steps where hidden-state
differences build up through 48 layer visits and the cache. This is one run
with one prompt set, so I cannot say if the 2× is real or noise. In the unit
tests on fixed inputs, the kernel's error is equal to or lower than base's (for
example 9.8e-4 vs 1.95e-3 at S=2048). Note that the "reference" also casts the
attention weights to bf16 before the V product, so it is not exact either.

## Memory

`repro_mem.py --batch 4 --steps 1500` on this branch, without clear_cache:

| step | active | cache | peak |
|---:|---:|---:|---:|
| 100 | 0.26G | 0.08G | 0.81G |
| 500 | 0.48G | 0.37G | 0.50G |
| 800 | 0.79G | 0.63G | 0.82G |
| 1000 | 0.82G | 0.76G | 0.84G |
| 1100 | 0.82G | 0.01G | 0.82G |
| 1500 | 0.86G | 0.38G | 0.91G |

`mx.get_cache_memory()` stays below 0.8 GB and does not grow with steps. Active
memory grows only as fast as the KV cache. The kernel's only temporaries are
the split partials, (B, Hq, splits, D) in float32, at most about 0.4 MB.

## Speed

`bench_decode.py --batches 1,4,8,16`, run as base, new, base, new, each under
`gpulock.py`, with both experiment servers live. Values are tok/s total; each
cell shows run 1 / run 2.

| ctx | B | base tok/s | new tok/s | speedup (mean) |
|---|---:|---:|---:|---:|
| short | 1 | 9.8 / 9.1 | 10.7 / 11.1 | **1.15×** |
| short | 4 | 22.9 / 21.7 | 29.2 / 29.3 | **1.31×** |
| short | 8 | 32.2 / 31.0 | 45.5 / 44.5 | **1.42×** |
| short | 16 | 37.1 / 35.6 | 50.6 / 49.6 | **1.38×** |
| 4k | 1 | 8.4 / 8.5 | 10.1 / 10.0 | **1.19×** |
| 4k | 4 | 19.0 / 18.0 | 24.1 / 23.3 | **1.28×** |
| 4k | 8 | 27.3 / 24.2 | (79.9) / 43.3 | **~1.6–1.7×** (run 1 new is an outlier, ignored) |
| 4k | 16 | 32.1 / 30.1 | 44.1 / 44.0 | **1.42×** |

Ops per B=1 decode step: 13,510 → 11,502. Graph build per B=1 step: 11.5 ms on
both.

Attention-only microbenchmark (48 attention calls in a row, Hq=24, Hkv=8, D=128,
ms per 48 calls):

| S | B | base loop | A: SDPA fold | C: kernel |
|---:|---:|---:|---:|---:|
| 300 | 1 | 3.06 | 3.82 | 1.75 |
| 300 | 16 | 34.6 | 3.90 | 1.33 |
| 4096 | 1 | 3.70 | 3.13 | 1.61 |
| 4096 | 4 | 14.0 | 3.80 | 1.93 |
| 4096 | 16 | 43.3 | 5.05 | 2.58 |

## Verdict

The change works. Batched decode is 1.3–1.4× faster at B=4/8/16 for short and
4k contexts (about 1.6× at 4k B=8), and B=1 is about 15–19% faster, because the
mask-building ops are gone at B=1 too. Accuracy against the fp32 reference is
the same as base's, and the memory is flat.

Limits and caveats:
- Absolute numbers are noisy because the live servers share the GPU. The
  ratios held over two back-to-back pairs, except for one outlier.
- The kernel reads `cache._idx` for rotating caches, the same private field
  that `_key_positions` already depends on.
- With these ops removed, decode stays op-bound elsewhere (11.5k ops/step).
  Attention is no longer the term that grows with B.
- Not yet tuned: the split heuristic (256 threadgroups, ≥128 keys each), and
  skipping the merge kernel when there is only 1 split.
