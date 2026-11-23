# Copyright © 2026 Apple Inc.
"""B=1 greedy speculative decoding of Inkling 2 with its DSpark draft.

Each round drafts `block_size - 1` tokens, verifies the anchor and the drafts
in one target forward, keeps the longest matching prefix plus the target's
next token, and rolls the target caches back to it.
"""

import time

import mlx.core as mx

from mlx_lm.models.cache import ArraysCache, CacheList, KVCache, RotatingKVCache
from mlx_lm.models.inkling2.layers import SlidingDecoderLayer


class SpecRotatingKVCache(RotatingKVCache):
    """A rotating cache that only takes multi-token updates, which keep the
    keys in temporal order, so the newest keys can be dropped."""

    def update_and_fetch(self, keys, values):
        return self._update_concat(keys, values)

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        if n:
            assert self._idx == self.keys.shape[2]
            self.keys = self.keys[..., : self._idx - n, :]
            self.values = self.values[..., : self._idx - n, :]
            self._idx -= n
            self.offset -= n
        return n


class SpecArraysCache(ArraysCache):
    """Keeps the conv input windows of the last forward, so the conv state
    after any prefix of its tokens can be taken."""

    def __init__(self, size):
        super().__init__(size)
        self.windows = [None] * size

    def keep(self, num_tokens: int, num_new: int):
        """Use the state after the first `num_tokens` of the `num_new` tokens
        of the last forward."""
        if num_tokens == num_new:
            return
        for i, w in enumerate(self.windows):
            n = self.cache[i].shape[1]
            self.cache[i] = w[:, num_tokens : num_tokens + n]


def make_spec_cache(model):
    window = model.args.text.sliding_window_size
    return [
        CacheList(
            (
                SpecRotatingKVCache(max_size=window)
                if isinstance(layer, SlidingDecoderLayer)
                else KVCache()
            ),
            SpecArraysCache(4),
        )
        for layer in model.model.language_model.layer_visits
    ]


def _rollback(cache, num_tokens: int, num_new: int):
    for c in cache:
        c[0].trim(num_new - num_tokens)
        c[1].keep(num_tokens, num_new)


def _forward(model, tokens, cache, aux_ids):
    aux = {i: None for i in aux_ids}
    logits = model(tokens, cache=cache, aux=aux)
    return logits, [aux[i] for i in aux_ids]


def prefill(model, prompt, cache, aux_ids=None, step=2048):
    """Returns the last logits and, with `aux_ids`, the features of all prompt
    positions."""
    feats = []
    for s in range(0, len(prompt), step):
        tokens = mx.array(prompt[s : s + step])[None]
        if aux_ids is None:
            logits = model(tokens, cache=cache)
        else:
            logits, f = _forward(model, tokens, cache, aux_ids)
            feats.append(f)
        mx.eval(logits, feats)
    if aux_ids is None:
        return logits[:, -1], None
    feats = [mx.concatenate(fs, axis=1) for fs in zip(*feats)]
    return logits[:, -1], feats


def greedy_generate(model, prompt, max_tokens, eos=()):
    """Target-only B=1 greedy decode, one token per forward. Returns tokens
    and decode seconds (prefill excluded)."""
    cache = model.make_cache()
    logits, _ = prefill(model, prompt, cache)
    y = mx.argmax(logits, axis=-1)
    mx.eval(y)
    out = [y.item()]
    t0 = time.perf_counter()
    while len(out) < max_tokens and out[-1] not in eos:
        logits = model(y[None], cache=cache)
        y = mx.argmax(logits[:, -1], axis=-1)
        mx.async_eval(y)
        out.append(y.item())
    return out, time.perf_counter() - t0


def speculative_generate(model, draft, prompt, max_tokens, eos=(), stats=None):
    """B=1 greedy speculative decode. Output tokens equal greedy decoding of
    the target, up to numerics of the 8-token verify forward."""
    aux_ids = draft.args.target_layer_ids
    block = draft.args.block_size
    cache = make_spec_cache(model)
    draft.reset()
    logits, feats = prefill(model, prompt, cache, aux_ids)
    draft.store_context(feats, 0)
    last = [f[:, -1:] for f in feats]
    y = mx.argmax(logits, axis=-1).astype(mx.int32)
    mx.eval(y)
    out = [y.item()]
    pos = len(prompt)
    accepted = []
    t0 = time.perf_counter()
    while len(out) < max_tokens and out[-1] not in eos:
        drafts = draft.draft(y, pos, last)  # (1, block - 1)
        tokens = mx.concatenate([y.reshape(1, 1), drafts], axis=1)
        logits, feats = _forward(model, tokens, cache, aux_ids)
        target = mx.argmax(logits[0], axis=-1)
        mx.eval(drafts, target, feats)
        d, t = drafts[0].tolist(), target.tolist()
        a = 0
        while a < block - 1 and d[a] == t[a]:
            a += 1
        accepted.append(a)
        new = d[:a] + [t[a]]
        # Stop inside the block as plain greedy decoding would.
        for k, tok in enumerate(new):
            out.append(tok)
            if tok in eos or len(out) >= max_tokens:
                new = new[: k + 1]
                break
        _rollback(cache, a + 1, block)
        draft.store_context([f[:, : a + 1] for f in feats], pos)
        last = [f[:, a : a + 1] for f in feats]
        y = mx.array([t[a]], mx.int32)
        pos += a + 1
    elapsed = time.perf_counter() - t0
    if stats is not None:
        stats["accepted"] = accepted
        stats["rounds"] = len(accepted)
    return out, elapsed
