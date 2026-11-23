# engine-fuse: fewer GPU ops per Inkling 2 Mini decode step

Branch `engine-fuse`, from `engine-base` (`d421048`). Final code: `3d75fec`.
All changes are in `mlx_lm/models/inkling2/layers.py`, and they are bit-exact with base.

## Result (real model, back to back in one GPU-lock session)

`bench_decode.py`, `pipenetwork/Inkling-2-Mini-MLX-bf16`. Base ran 21:32–21:35, then fuse ran 21:35–21:38, in the same lock hold
(`bench/runs/final_base.txt`, `bench/runs/final_fuse.txt`).

| | base | fuse | change |
|---|---:|---:|---:|
| ops / decode step (B=1, export_to_dot) | 13510 | 9468 | −30% |
| graph build / step (B=1) | 11.5 ms | 10.1 ms | −12% |
| short ctx, B=1 | 107.4 ms | 93.4 ms | −13% (9.3 → 10.7 tok/s) |
| short ctx, B=4 | 191.9 ms | 143.8 ms | −25% (20.8 → 27.8 tok/s) |
| short ctx, B=8 | 258.7 ms | 184.5 ms | −29% (30.9 → 43.4 tok/s) |
| 4k ctx, B=1 | 122.1 ms | 103.0 ms | −16% (8.2 → 9.7 tok/s) |
| 4k ctx, B=4 | 232.8 ms | 187.8 ms | −19% (17.2 → 21.3 tok/s) |
| 4k ctx, B=8 | 307.5 ms | 229.6 ms | −25% (26.0 → 34.8 tok/s) |

The two live experiment servers share the GPU, so the absolute numbers are noisy.
Two earlier pairs pointed the same way, but their two halves were 15–25 min apart in the lock queue:
- **r1** (only the first commit): ops 13510 → 10790. Fuse was faster at all six points, by about 5–30%. Base's short B=1 reading (140 ms) was an outlier.
- **r2** (all commits, including the one later reverted): ops 13510 → 8808, and 13–26% faster.

The gain grows with batch size. A large part of the cut is in per-row work: the attention mask was built once per row, and integer indexing made copies.

## Correctness

- **Real model** (`bench/correctness.py`, base vs `3d75fec`):
  - Logprobs of the first 32 greedy steps for 3 prompts: max abs diff **0**.
  - Greedy tokens over 200 steps at B=1 (3 prompts): **identical**.
  - `batch_generate` B=4 with 4 different prompts, 200 tokens each: **identical** text.
  - Run log: `bench/runs/correct_c2.txt`.
- **PR tests:** `test_inkling2.py`, 8 tests OK.
- **Small random model:** this is a copy of the real config with the same 24 layers, the same 6–18 ×3 loop and the same dtypes, but small widths (`bench/opcount.py --dump`). Base and fuse logits are bit-identical over 300 decode steps, at B=1 and B=3.
- **Each compiled helper:** each compared with the same code run eagerly under `mx.disable_compile()`, on 1–4M random elements: 0 differences (`bench/exact_fns.py`). Each change also keeps the exact op order and the intermediate bf16 roundings of the old code. For example, the conv residual still divides by `sqrt(K+1)` and does not multiply by the inverse.

## What each change did

Op counts are from the small replica (`bench/opcount.py`): ops are computed with real-model structure, so the counts track the real model's (13270 vs 13510 for base). "Kernel ops" leaves out view-only nodes (Reshape, Slice, Broadcast, Transpose, …), so it is about the number of kernel launches. "batch" uses the `BatchGenerator` caches, the path the server uses. The ms columns are from the real model, measured cumulatively at r1 and in the final run. Per-commit log: `bench/runs/opcount_per_commit.log`.

| commit | change | batch B=1 kernel ops | batch B=4 kernel ops |
|---|---|---:|---:|
| base | – | 7557 | 10005 |
| `12a548a` | `mx.compile(shapeless)` for: attention output gate; tau scaling fused into the q/rel L2 clip; short-conv residual tail; MessageGate tanh mix; f32 residual adds; MoE weighted sum. The decode attention mask is built once for the whole batch in one compiled function, not once per row. The decode conv state is a view of the 4-row window, not an arange plus a gather. | 5213 | 5549 |
| `39d29a4` | RMS norm of the values on bf16 directly (the kernel computes in f32 already, so the result is bit-identical), which drops 2 casts per layer. One matmul for each MessageGate's forget and message projections (L=1 only). | 5021 | 5357 |
| `b8c6acb` | One matmul for the q/r/gate/sink(+k/v) projections (L=1). **Reverted in `3d75fec`**, see below. | (4837) | (5173) |
| `8430933` | The q/k/v message gates of a looped layer run as one batched RMS norm / matmul / mix (L=1). | 4529* | 4865* |
| `53afff2` | `x[i]` on the leading axis makes a Gather (a copy kernel) in MLX. It is replaced by a slice view (masks, sinks, the message-input query, gate outputs). | 4385* | 4577* |
| `77fe1a9` | The position aranges (and the rotated key positions of the sliding cache) are made once per step and shared by all 48 layer visits. | 4201* | 4393* |
| `dff0678` | Dynamic short conv (full layers): static weight laid out once; the cast, add and multiply fused; residual tail fused. | 4089* | 4281* |
| `3d75fec` | Revert of `b8c6acb`. | **4273** | **4465** |

\* These rows still include `b8c6acb`. Without it they are about 184 ops higher, as the final row shows.

Net: **−43% kernel ops at B=1 and −55% at B=4** on the batch path. On the bench's ops/step metric (real model, plain cache, all nodes counted) the drop is −30%.

## What didn't work, or was dropped

- **Merged attention-projection matmul (`b8c6acb`), reverted.** In a micro-test with random inputs it was bit-exact at M=1/4. On the real model, the greedy text of prompt 0 split from base at token 147. A full-length dump located the first logprob difference at step 112, then a bisect over the commits on the real model pinned it to this commit; base and fuse are each deterministic run to run. So the wider GEMV rounds differently on some real activations. It saved about 4% of kernel ops (r2's slightly faster numbers include it), but it was not bit-exact, so it is out. It is also not used for prefill, where the GEMM tiling changes the rounding even with random inputs.
- **Elementwise depthwise conv for L=1, not committed.** It was bit-exact with `conv1d`, but `mx.compile` split it into 3 kernels plus 16 slice/squeeze nodes per call. That is worse than `conv1d` plus one fused residual kernel, so I kept `conv1d`.
- **Graph-build time** barely moved (11.5 → 10.1 ms). The step is GPU bound, not Python bound: each compiled call costs some Python time, which offsets the nodes it removes. This is fine while the build stays hidden behind the GPU step under `async_eval`.
- **Not tried:** compiling a whole decoder layer for L=1. Most of what is left is not elementwise (RMSNorm, matmul, concatenate for the conv window, custom kernels, SDPA, KV-cache SliceUpdate), so `mx.compile` would fuse little more. It would also have to keep cache mutation out of the compiled region and retrace for every B.

## What is left (ideas, not done)

- About 730 casts remain. Most are `nn.RMSNorm` on f32 input followed by `.astype(bf16)` before a matmul, the cast of the concatenated MessageGate input to bf16, and the router's f32 cast. Removing them exactly would need a custom kernel, for example an RMS norm that reads f32 and writes bf16.
- `take_along_axis` in the decode mask makes 2 aranges plus a gather per visit (about 150 ops).

## Constraints followed

- I worked only in this worktree and committed only to `engine-fuse`; nothing was pushed.
- Every real-model load and benchmark ran under `common/gpulock.py`. The fuse runs used a `git archive` snapshot of a commit, so edits could go on while a run waited for the lock (`bench/bench_pair.sh`, `bench/correct_pair.sh`).
- No new S-growing f32 temporaries were added. The decode mask stays in bf16, and its intermediates stay inside the fused kernel.
- The small-model op counts, dumps and helper tests ran without the lock. They use a model of a few MB for a few seconds, like the unit tests do.
