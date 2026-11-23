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

## Verification

Filled in below by the checks run after the merge.
