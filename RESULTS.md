# engine-combined: fusion + batched decode kernel

Local experiment branch, not for upstream submission. It merges:

- `engine-fuse` (code at `3d75fec`): `mx.compile` fusion and fewer ops per
  decode step. Bit-exact with base. See `git show engine-fuse:RESULTS.md`.
- `engine-batched-sdpa` (`7973ed2`): one Metal kernel (`decode_attention`)
  for every decode step, any batch size. See
  `git show engine-batched-sdpa:RESULTS.md`.

Merge resolution: both branches rewrote the `L == 1` path of
`Attention._attention`. The kernel now serves every GPU decode step, so
fusion's per-row SDPA loop and its `_decode_mask` helper were removed. The CPU
path falls back to `_attention_reference`, as in `engine-batched-sdpa`.

## Verification (real model, `verify_all.sh`, one GPU-lock hold, P5 sweep live on the GPU)

- PR tests (`test_inkling2.py`): 8/8 OK, after limiting the decode kernel to
  `head_dim % 32 == 0` (it splits a head over 32 lanes; the tests' tiny model
  has head_dim 16 and failed 3 tests on `engine-batched-sdpa` too). Other head
  sizes use `_attention_reference`.
- `test_decode_attention.py`: all cases OK, including rotating caches after wraparound.
- Teacher-forced B=1 over base's greedy tokens (4 prompts x 200 steps),
  combined vs base: top-1 agree 793/800; base top1-top2 margins at the flips
  0.0, 0.03, 0.10, 0.11, 0.17, 0.80, 4.87. Max |d logprob(ref token)| 4.12.
  For scale: base vs the fp32 reference differs by up to ~7 in raw logits
  (engine-batched-sdpa RESULTS.md), so this is within existing bf16 noise.
- Greedy B=4 batch, 200 tokens: 2 of 4 prompts identical, the others split
  at tokens 52 and 32.
- Memory, `repro_mem.py --batch 4 --steps 1500`: buffer cache <= 0.8 GB, flat.

| ctx | B | base tok/s | combined tok/s | speedup |
|---|---:|---:|---:|---:|
| short | 1 | 14.6 | 16.6 | 1.14x |
| short | 4 | 31.8 | 45.0 | 1.42x |
| short | 8 | 56.0 | 72.1 | 1.29x |
| short | 16 | 57.5 | 72.4 | 1.26x |
| 4k | 1 | 14.6 | 16.0 | 1.10x |
| 4k | 4 | 29.5 | 35.3 | 1.20x |
| 4k | 8 | 41.8 | 55.0 | 1.32x |
| 4k | 16 | 44.1 | 57.7 | 1.31x |

Ops per B=1 decode step: 13,510 -> 8,555 (-37%). The gains do not multiply:
much of fusion's win was per-row mask work, which the kernel also removes.
Raw output: `verify/` and the run log.

## Deployed

Installed into `~/.local/share/uv/tools/mlx-lm/.../mlx_lm/models/inkling2/`
(`layers.py`, `kernels.py`), with `*.engine-base` backups next to them.
Rollback: copy the two `.engine-base` files back and restart the servers.
