# Inkling 2 Mini DSpark draft on MLX: speculative decoding

Branch `engine-drafter` (from `engine-base`). M4 Max 128 GB, target
`pipenetwork/Inkling-2-Mini-MLX-bf16`, draft `mtp.safetensors` from
`thinkingmachines/Inkling-2-Mini` (put in `weights/`, not committed).

## Reference

The shipped `mtp.safetensors` is a **DSpark** draft (Thinking Machines call
the target-feature mixing "DFlare"). The reference used is vLLM's Inkling 2
support (vllm-project/vllm `main`, commit `abb8194` for the draft file):

- `vllm/models/inkling2/nvidia/dspark.py`: `Inkling2DSparkModel` (the draft)
- `vllm/models/inkling2/nvidia/ops/dspark.py`: its kernels (mix + RMSNorm,
  q/k norm + RoPE, Markov gate, add + softcap)
- `vllm/models/inkling2/speculator.py`: `Inkling2DSparkSpeculator`
  (f_rnn, sequential Markov sampling)
- `vllm/models/inkling2/configs.py`: `Inkling2DSparkConfig` (block layout)
- `vllm/models/inkling2/nvidia/loop.py`, `model.py`: where target features
  are captured (`aux_hidden_state_layers`)
- `vllm/v1/worker/gpu/spec_decode/{dspark,dflash}/speculator.py`: the generic
  DSpark/DFlash loop (query positions, sample indices, context K/V)

The vLLM day-0 Inkling 2 Small post serves the model with a DSpark speculator
and 7 speculative tokens. I did not find a transformers implementation of
the draft. The HF model card says nothing about it.

## Architecture (as implemented in `mlx_lm/models/inkling2/dspark.py`)

Target features. `target_layer_ids = [5, 23, 41, 48]` are *layer visits*
(loop passes counted, 48 visits). For ids < 48 the feature is the **input** of
that visit: the attention-residual partial block (the `hidden_states` that
goes into `DecoderLayer`). Visit 5 = layer 5 (prelude), 23 = layer 11 in loop
pass 2, 41 = layer 17 in loop pass 3. Id 48 is the target's final-norm output
(the input of `lm_head`). So 4 features of 2304 per token.

Mix ("DFlare"). `fuse_alpha = softmax(fuse_weight, -1)` (4 rows x 4 taps).
Row i (draft layer i) or row 3 (Markov head): `RMSNorm_w(sum_t alpha[row,t] *
feat_t)` in fp32, with that layer's `dflare_norm` weight.

Block layout (fill-in, `block_size` 8). Queries are the anchor (the last
committed token, embedded with the target's `embed_norm(embed_tokens)`) at
position p and 7 `mask_embed` vectors at p+1..p+7. Mask j predicts the token at
its own position p+j. So one block gives 7 draft tokens.

Draft backbone: 3 pre-norm dense layers (hidden 2304, 24 q heads / 8 kv heads,
head_dim 128, SwiGLU 9216, RMSNorm eps 1e-6, neox RoPE theta 1e6).
- q, k: per-head RMSNorm (`q_norm`, `k_norm`), RoPE at the block positions.
- Context K/V per draft layer, from the mixed features of committed positions
  0..p-1: `k = RoPE(k_norm_ctx(k_proj_ctx(mix_i)))`, `v = v_proj_ctx(mix_i)`.
  They are computed once per committed token and cached (the draft KV cache).
- Attention is **not causal**: every block query sees all context keys and all
  8 block keys. Layers 0 and 2 (`local_layer_ids`) see only the last 1024
  context positions.
- Final `norm`. Base logits = target `lm_head` (shared) of the 7 mask rows.

Markov head (`rnn_hidden`, rank 256), run left to right over the 7 positions:
- `f_rnn = mix_3(features of position p-1)` (the token before the anchor).
- `rnn_proj` (768 x 5120) input is `[state, markov_w1[prev], hidden_j, f_rnn]`.
  The `[hidden_j, f_rnn]` part is computed for all 7 positions at once.
- `gate, cand, out = split(raw)`; `state = sig(gate)*state +
  (1-sig(gate))*tanh(cand)`; `latent = tanh(out)` (state starts at 0, `prev`
  starts at the anchor).
- `bias = lm_head(hidden_out(latent))`, folded once into a (vocab x 256)
  matrix `lm_head.W @ hidden_out.W` plus `lm_head.W @ hidden_out.b`.
- `logits = 30 * tanh((base_j + bias) / 30)`; greedy `prev = argmax`.

The draft uses the target's `embed_tokens`, `embed_norm` and `lm_head`; the
636 MB file has only its own layers, `markov_w1` (vocab x 256) and the norms.

## What was built

- `mlx_lm/models/inkling2/model.py`: `Model(..., aux={visit_id: None})` fills
  in the target features (the input of the given layer visits, and id 48 = the
  final hidden). Without `aux` the forward is unchanged; a test checks that the
  logits are bit-identical with and without `aux`.
- `mlx_lm/models/inkling2/layers.py`: when a conv cache has a `windows` list,
  the layer also keeps the conv input windows of the last forward (3 lines).
- `mlx_lm/models/inkling2/dspark.py`: the draft (`load_draft(target, "weights")`).
  The draft graph is lazy end to end (the 7 Markov argmaxes feed the next step
  without a sync), so one `mx.eval` per round covers the draft and the verify.
- `mlx_lm/models/inkling2/speculative.py`: B=1 greedy loop.
  Each round: draft (anchor at p) -> one target forward of `[anchor, d1..dn]`
  with feature capture -> accept the longest prefix where `d_k == argmax_{k-1}`
  -> commit it plus the target's next token -> roll back -> add context K/V of
  the committed positions to the draft cache. Rollback:
  - full layers (`KVCache`): `trim`.
  - sliding layers: `SpecRotatingKVCache`, which takes every update through
    the concat path. That keeps the keys in temporal order (at most
    1023 + L keys), so dropping the newest n keys is a slice. (The stock
    `RotatingKVCache` is not trimmable once the window is full, so
    mlx-lm's `speculative_generate_step` cannot be used past 1024 tokens.)
  - the 4 short-conv states per visit (`SpecArraysCache`): the state after k of
    the L new tokens is `window[:, k:k+3]` of the kept conv input window.
  - `num_draft` (1..7) sets how many drafts a round verifies.
- `spec_bench/`: `run.py` (offline acceptance, spec vs greedy), `cost.py`
  (forward cost per L), `diag.py` (divergence analysis), `test_spec.py`
  (tiny-model test: with a fake drafter at 0/15/50/100% wrong drafts and 1/3/7
  drafts per round, speculative output == greedy, through the sliding
  window and loop). Logs: `spec_bench/run1.log`, `run2.log`.

Tests: `common/test_inkling2.py` passes (`OK`), `spec_bench/test_spec.py` passes.

## Acceptance

Offline (teacher forced on target greedy generations, 3 chat prompts at
reasoning_effort=low, 160 tokens, 419 anchors): **mean 2.06 accepted drafts per
7-token block = 3.06 tokens per round.** Histogram of accepted 0..7:
`[112, 97, 78, 46, 28, 18, 9, 31]`. Per prompt: 2.22 (code), 1.40 (short
explanation), 2.39 (algorithm).

Ablations as wiring checks (same anchors):

| variant | mean accepted |
|---|---|
| reference wiring | **2.06** |
| features taken one visit later (output of visits 5/23/41) | 1.72 |
| Markov head off (base logits only) | 0.79 |

Online, in the speculative loop (6 prompts, 256 tokens max): mean 1.75 accepted
of 7 (2.75 tokens/round), and 1.41 of 3 (2.41 tokens/round) with
`num_draft=3`. Per prompt 0.96 (short story) to 3.22 (code refactor).

## Speed (B=1, greedy, M4 Max shared with the two live experiment servers)

All numbers are with the live runs on the GPU, so a base B=1 step is about
100 ms (9-10 tok/s), not the unloaded speed. Base and speculative ran back to
back in one process; the base env `bench_decode.py` ran right after.

Target forward cost by number of tokens (short context, `cost.py`):

| L | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| ms | 102 | 128 | 165 | 217 |

Draft block: 13.5 ms (backbone 4.3 ms, the 7 sequential Markov steps with the
folded vocab projection about 9 ms).

So an 8-token verify costs **2.1x** a 1-token step here, not ~1x. A likely
cause (not profiled): with bf16 experts, 8 tokens route to many more distinct
experts per visit than 1 token, so the MoE reads more weight memory. The
earlier "B=4 costs about B=1" result may not carry over to L=8 of one sequence
on this loaded GPU.

End to end (`run2.log`, 6 prompts, 256 tokens, aggregate tok/s):

| | tok/s | vs base |
|---|---|---|
| base env `bench_decode.py` B=1 short ctx | 9.4 | |
| target greedy (same process, back to back) | 10.2 | 1.00x |
| speculative, `num_draft=7` (verify 8) | 12.2 | **1.19x** |
| speculative, `num_draft=3` (verify 4) | 14.6 | **1.43x** |

An earlier run (`run1.log`, `num_draft=7`) gave 12.6 vs 9.5 tok/s (1.33x).
Per prompt with `num_draft=3`: 1.11x (story) to 1.81x (code). With
`num_draft=7` the story prompt was slower than base (0.84x).

## Exactness

Speculative output was token-identical to target greedy on 2 of 6 prompts (both
120-token answers). On the other 4 it matched for 36-184 tokens, then diverged.
`diag.py` shows these are bf16 numerics of multi-token forwards, not a
rollback bug:

- At every divergence the greedy top-1/top-2 logit gap is tiny: 0.15, 0.099,
  0.000, 0.271 (median gap per prompt 1.7-9.4).
- A one-shot stock prefill of the same text (stock caches, no speculative code)
  also disagrees with token-by-token greedy at near-tie positions (gaps
  0.03-0.56).
- Teacher-forced 8-token chunks through the speculative caches give exactly the
  speculative divergence points (52, 184, 161, 36). With `num_draft=3` (4-token
  chunks) the first divergence moves (e.g. 52 -> 147), as chunk-size numerics
  would.
- With exact arithmetic (tiny fp32 model, `test_spec.py`) the output is
  identical to greedy for every acceptance pattern.

So the loop gives the target's own greedy output up to the same near-tie
reorderings a batched prefill shows. It is not bit-exact against 1-token
decode. A bit-exact version would need the L=1 and L>1 paths to share one
kernel order (MoE gather matmul, attention). I did not try that.

## Verdict

Feasible and working: the shipped DSpark draft runs in MLX, faithful to the vLLM
reference (ablations confirm the wiring). It gives **~1.2-1.45x B=1 decode
speed** on this shared machine, best with `num_draft=3`. The gain is less than
the acceptance suggests, for two reasons:

1. The premise "verifying 8 tokens costs about one step" does not hold here:
   L=8 costs 2.1x L=1, L=4 1.6x (the MoE expert reads grow with the number of
   tokens). That is why verifying fewer drafts wins.
2. Acceptance on this model is moderate (2.06 / 7 offline; 1.75 online), and
   low on free prose.

Dead ends and limits:
- No transformers reference for the draft was found; vLLM was the only one.
- Speed numbers are on a GPU shared with live runs. The L-cost ratios, and so
  the best `num_draft`, may change on an idle machine. Re-measure with
  `cost.py` then.
- B=1 only. Batched verify is not built. Rejection sampling (temperature > 0)
  is not built: greedy only.
- The draft's Markov loop is 7 sequential vocab-size projections (~9 ms per
  round). A top-k variant (vLLM has `_sample_sequential_topk`) or a fused
  kernel could cut that, at some change in the draft distribution.
- Not integrated into `mlx_lm.server`/`generate`; use
  `speculative_generate(model, load_draft(model, "weights"), prompt_ids, n, eos, num_draft=3)`.
- Prefill with feature capture runs in 2048-token chunks; only short prompts
  were benchmarked (4k context not measured with the draft).
