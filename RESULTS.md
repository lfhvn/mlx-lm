# Self-speculative decoding on the variable-loop checkpoint

Model: `/Users/lhm/models/TML-d24-varloop-mlx` (nvfp4, looped core = layers 6–17,
post-trained at 3 passes). Target = 3 passes. Draft = the same weights at 1 or 2
passes. B=1, greedy, M4 Max, with the two experiment servers running. All GPU
runs went through `gpulock.py`. Plain and speculative runs were made back to back
in the same process.

## Verdict

**There is a real speedup, but it is modest: about 1.25× at short context and
1.15× at 4k, with a 1-pass draft, k = 2–4, and the "shared passes" variant
below.** The naive version (a separate draft cache, verify recomputes all
3 passes) gives only about 1.14× at short context and is **slower** at 4k
(0.85×). A 2-pass draft is never worth it (≤ 1.10×). k ≥ 5 loses.

It is not the 2–3× that "verify is nearly free" suggests, for two reasons:

1. **Verify is not nearly free on this model.** One 3-pass forward of
   L = 1/2/4/7 tokens costs 85/103/143/193 ms at short context and
   80/140/195/231 ms at 4k. So 4 tokens cost 1.7× (short) to 2.4× (4k) as much
   as 1 token. The op count is the same, but each token routes to its own
   8 of 128 experts, so the MoE reads more weights as L grows. The multi-query
   attention kernel also scales badly at 4k for small L.
2. **The draft is not cheap.** A 1-pass forward is 24 of 48 layer visits, and
   it costs 47 ms against 85 ms for 3 passes (0.56×; the lm_head, embedding and
   launches do not shrink). A 2-pass draft is 0.73×.

Acceptance is good. A 1-pass draft agrees with the 3-pass model on 87% of
tokens and gets about 2.4 of 3 drafts accepted. A 2-pass draft agrees on 93%
and gets 2.7 of 3, but its higher cost cancels the gain.

## Two implementations

* `spec` (`loopspec/spec.py: spec_greedy`): one model object. The pass count
  is switched per call with `args.loop_config["num_passes"]`. Draft and target
  each have their own cache (24 or 36 visits for the draft, 48 for the target).
  The target verifies `[y, d1..dk]` in one forward.
* `shared` (`loopspec/shared.py: SharedSpec`): uses the loop structure. Pass j
  reads only the prelude output and pass j-1, so the prelude and passes 1..dp
  of the 3-pass target are **the same computation** as the dp-pass draft. Draft
  and target share those caches and activations. The draft adds its own coda
  (layers 18–23). Verify runs the shared part for d_k only (one token), then
  passes dp+1..3 and the target coda on all k+1 tokens. For dp = 1, verify
  runs 30 of 48 visits at L = k+1 instead of 48. This also removes the second
  prefill. Check (`shared_smoke.py`): after prefill, the target and draft
  logits from `SharedSpec` equal the normal 3-pass and dp-pass forwards
  exactly (max |diff| = 0.0).

mlx-lm's `speculative_generate_step` could not be reused. It needs a separate
`draft_model` and trimmable caches. This model's short-convolution state
(`ArraysCache`) cannot be trimmed, and a `RotatingKVCache` cannot be trimmed
after 1024 tokens (rolling back a ring buffer loses the evicted rows). So the
code adds:
* `WindowKVCache`: a linear buffer for the sliding layers. It keeps the last
  window−1+32 tokens, so it can be trimmed. Positions and window masking work
  unchanged.
* `WindowConvCache` and a 3-line hook in `DecoderLayer`
  (`mlx_lm/models/inkling2/layers.py`): keep the conv input windows of the
  last call, so the state can be rolled back to any token of a verify block.
  The draft restores per-step snapshots instead.

## (a) Agreement under teacher forcing

The 3-pass model's own greedy decode is the reference: 256 tokens from each of
5 coding prompts (`loopspec/prompts.py`). Then the whole sequence is run in one
forward at N passes and the argmax at each position is compared.
(`loopspec/agree.py`, `loopspec/out/agree.json`)

| prompt | tokens | 1 pass | 2 passes | 5 passes | 3 passes (TF vs decode) |
|---|---|---|---|---|---|
| 0 | 256 | 88.3% | 94.5% | 95.7% | 98.4% |
| 1 | 256 | 84.4% | 91.4% | 91.8% | 96.5% |
| 2 | 256 | 91.4% | 95.3% | 97.3% | 99.6% |
| 3 | 256 | 87.5% | 94.1% | 96.1% | 99.2% |
| 4 | 256 | 84.4% | 89.5% | 94.5% | 97.3% |
| **all** | 1280 | **87.2%** | **93.0%** | 95.1% | 98.2% |

The last column is the noise floor. The same 3-pass model run as one
multi-token forward instead of one token per step already differs on 1.8% of
tokens, all of them near-ties (bf16/nvfp4 numerics).

Acceptance predicted from these match vectors (`loopspec/simulate.py`) agrees
with what was measured: 1 pass, k = 3: 2.26 accepted per round (measured
2.37–2.54); 2 passes, k = 3: 2.54 (measured 2.66–2.77).

## (b)/(d) Speed at B=1

Decode tok/s, timed from the first generated token (prefill excluded), 128
tokens per run. "runs" = prompt runs averaged. Speedup = spec / plain from the
same process, measured right before it. Short context = prompts of 34–55
tokens. 4k = 4000 filler tokens + a coding question.
(`loopspec/bench.py`, raw: `loopspec/out/bench_*.json`, logs `log_*.txt`)

| ctx | variant | draft passes | k | runs | speedup mean (min–max) | spec tok/s | plain tok/s | accepted/k | tok/round |
|---|---|---|---|---|---|---|---|---|---|
| short | **shared** | 1 | 2 | 5 | **1.24× (1.11–1.46)** | 13.7 | 11.0 | 1.65/2 | 2.64 |
| short | **shared** | 1 | 3 | 5 | **1.24× (1.10–1.43)** | 13.6 | 11.0 | 2.37/3 | 3.34 |
| short | **shared** | 1 | 4 | 3 | **1.28× (1.17–1.40)** | 14.7 | 11.5 | 3.23/4 | 4.18 |
| short | shared | 1 | 5 | 2 | 1.10× (1.10–1.10) | 11.3 | 10.3 | 3.39/5 | 4.34 |
| short | shared | 2 | 2 | 3 | 1.07× (1.02–1.14) | 11.2 | 10.4 | 1.85/2 | 2.83 |
| short | shared | 2 | 3 | 3 | 1.07× (1.06–1.08) | 11.2 | 10.4 | 2.66/3 | 3.66 |
| short | shared | 2 | 4 | 3 | 1.05× (1.01–1.09) | 11.0 | 10.4 | 3.50/4 | 4.47 |
| short | spec | 1 | 2 | 8 | 1.14× (1.05–1.17) | 12.7 | 11.2 | 1.66/2 | 2.65 |
| short | spec | 1 | 3 | 8 | 1.13× (1.02–1.19) | 12.7 | 11.2 | 2.40/3 | 3.37 |
| short | spec | 1 | 4 | 6 | 1.16× (1.06–1.50) | 13.3 | 11.5 | 3.05/4 | 4.00 |
| short | spec | 1 | 5 | 5 | 0.99× (0.90–1.05) | 10.9 | 11.0 | 3.27/5 | 4.20 |
| short | spec | 1 | 6 | 3 | 0.95× (0.84–1.08) | 11.0 | 11.5 | 3.70/6 | 4.58 |
| short | spec | 2 | 2 | 3 | 1.02× (0.93–1.18) | 9.6 | 9.5 | 1.86/2 | 2.85 |
| short | spec | 2 | 3 | 3 | 1.04× (0.94–1.20) | 9.8 | 9.5 | 2.77/3 | 3.73 |
| short | spec | 2 | 4 | 3 | 1.01× (0.93–1.13) | 9.5 | 9.5 | 3.65/4 | 4.58 |
| short | spec | 2 | 5 | 3 | 0.91× (0.73–1.14) | 8.5 | 9.5 | 3.99/5 | 4.86 |
| short | spec | 2 | 6 | 3 | 0.91× (0.71–1.16) | 8.5 | 9.5 | 4.82/6 | 5.63 |
| 4k | **shared** | 1 | 2 | 1 | **1.15×** | 10.2 | 8.9 | 1.71/2 | 2.67 |
| 4k | **shared** | 1 | 3 | 1 | **1.13×** | 10.1 | 8.9 | 2.46/3 | 3.46 |
| 4k | shared | 2 | 2 | 1 | 1.04× | 9.3 | 8.9 | 1.89/2 | 2.84 |
| 4k | shared | 2 | 3 | 1 | 1.10× | 9.8 | 8.9 | 2.85/3 | 3.76 |
| 4k | spec | 1 | 2 | 2 | 0.88× (0.87–0.89) | 8.5 | 9.7 | 1.43/2 | 2.42 |
| 4k | spec | 1 | 3 | 2 | 0.85× (0.83–0.87) | 8.2 | 9.7 | 1.95/3 | 2.91 |
| 4k | spec | 1 | 4 | 1 | 0.83× | 8.7 | 10.4 | 2.45/4 | 3.37 |
| 4k | spec | 2 | 2 | 2 | 0.83× (0.83–0.83) | 8.0 | 9.7 | 1.78/2 | 2.78 |
| 4k | spec | 2 | 3 | 2 | 0.82× (0.82–0.83) | 8.0 | 9.7 | 2.58/3 | 3.56 |
| 4k | spec | 2 | 4 | 1 | 0.82× | 8.6 | 10.4 | 3.33/4 | 4.27 |

Noise: plain decode alone moved between 8.5 and 11.7 tok/s across runs, because
of the live servers. Treat single-run differences below about 0.1× as noise.
The 1.50× spec run (k = 4, prompt 0) is an outlier. At 4k there is one long
prompt, so those rows are 1–2 runs.

Forward cost (ms, `loopspec/out/log_4k.txt`, and the `--micro` part of the
first short run):

| ctx | passes | L=1 | L=2 | L=4 | L=7 |
|---|---|---|---|---|---|
| short | 1 | 47 | 58 | 72 | 92 |
| short | 2 | 61 | 72 | 99 | 151 |
| short | 3 | 85 | 103 | 143 | 193 |
| 4k | 1 | 33 | 49 | 69 | 140 |
| 4k | 2 | 57 | 92 | 158 | 181 |
| 4k | 3 | 80 | 140 | 195 | 231 |

**B=4:** not done. mlx-lm's batch path (`BatchGenerator`) has no speculation,
and per-row acceptance needs ragged rollback across the batch caches. The
numbers above also predict a worse result at B=4: decode becomes less
launch-bound as B grows, and verify already scales poorly with token count.

Tried and dropped: running short verify blocks (L ≤ 8) through the decode SDPA
once per query instead of the prefill attention kernel. It was slower at every
L, short and 4k (diff saved in `loopspec/out/rows_attempt.diff`, timings in
`log_micro2.txt`).

## (c) Exactness

* **Rollback is bit-exact.** `loopspec/rollback_test.py`: 40 verify rounds with
  a random accept pattern, where the rejected tail is either the true text or
  random tokens. The logits of all kept positions match: max |diff| = 0.0 over
  119 positions. So rejected drafts leave no trace in the KV caches or the
  conv state.
* **Token-for-token equality with plain greedy holds only up to the first
  near-tie.** Speculative output equals plain 3-pass greedy (one token per
  step, stock cache) for 10–120 tokens, then changes at one token. At every
  divergence, plain greedy's own top-1 minus top-2 logit margin is < 0.09
  (three are exact ties at 0.0). The cause is numerics: the target computes the
  verify block as a k+1-token forward. Its logits differ from one-token
  forwards by about 0.04 (see `loopspec/diverge.py`: one-token,
  2/4-token and 8-token forwards give 27.495/27.495/27.455 for the same top
  token). The same effect makes 3-pass teacher forcing disagree with 3-pass
  decode on 1.8% of tokens (table (a)). This is not a speculation bug, and any
  batched verify on this bf16/nvfp4 model has it.

`loopspec/exact.py`, 128 tokens per prompt, k = 3 (first differing token index;
plain's top-2 margin there):

| prompt | spec, 1 pass | shared, 1 pass | shared, 2 passes | plain steps with margin < 0.1 |
|---|---|---|---|---|
| 0 | 54 (0.081) | 30 (0.041) | 92 (0.051) | 7 |
| 1 | 80 (0.024) | 80 (0.024) | 81 (0.034) | 4 |
| 2 | 72 (0.057) | 121 (0.035) | identical | 3 |
| 3 | 74 (0.000) | 120 (0.000) | 58 (0.000) | 7 |
| 4 | 57 (0.049) | 93 (0.041) | identical | 7 |

Each emitted token is the 3-pass target's argmax given the emitted prefix (as
computed by the verify forward). So the output is valid 3-pass greedy text, but
it is not bit-identical to the one-token-per-step decode.

## Ways to make this faster (not done)

* Draft cost is the limit. With the shared variant, a 1-pass draft step still
  runs the full lm_head (201k × 2304) and the coda. A draft that skips the coda
  and reads logits from the pass-1 hidden state (no trained head for that) or
  from the MTP head (`mtp_config` exists) could be much cheaper.
* Tree or multi-candidate verify will not help: verify cost grows with L
  through the MoE.
* Fusing the draft loop (`mx.compile`) would cut launch overhead, which is a
  large part of the 47 ms.

## Files

* `loopspec/spec.py`: caches with rollback, plain greedy, `spec_greedy`.
* `loopspec/shared.py`: `SharedSpec` (shared prelude and early passes).
* `loopspec/agree.py`, `simulate.py`: (a). `bench.py`, `summarize.py`:
  (b)/(d). `exact.py`, `diverge.py`, `rollback_test.py`, `shared_smoke.py`:
  (c) and correctness. `micro2.py`: the dropped attention experiment.
* `mlx_lm/models/inkling2/layers.py`: a 3-line hook (`keep_windows`) with no
  effect unless a cache asks for it.
* Raw outputs: `loopspec/out/`.
