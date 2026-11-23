"""Self-speculative decoding for the variable-loop Inkling checkpoint.

One model object, one set of weights. The draft runs the looped core
(layers 6-17) `draft_passes` times, the target runs it `target_passes` times.
Each has its own cache, because the number of layer visits differs.
"""

import time

import mlx.core as mx

from mlx_lm.models.cache import ArraysCache, CacheList, KVCache, make_prompt_cache
from mlx_lm.models.inkling2.layers import SlidingDecoderLayer


def text_model(model):
    return model.model.language_model


def set_passes(model, passes: int):
    text_model(model).args.loop_config["num_passes"] = passes


class WindowKVCache:
    """Linear (not ring) KV cache for sliding layers. It keeps the last
    `window - 1 + slack` tokens, so it can be trimmed by up to `slack`."""

    def __init__(self, window: int, slack: int = 32, step: int = 256):
        self.window, self.slack, self.step = window, slack, step
        self.keys = self.values = None
        self.offset = 0
        self.idx = 0

    def update_and_fetch(self, keys, values):
        S = keys.shape[2]
        if self.keys is None or self.idx + S > self.keys.shape[2]:
            keep = min(self.idx, self.window - 1 + self.slack)
            B, H, _, D = keys.shape
            cap = keep + S + self.step
            nk = mx.zeros((B, H, cap, D), keys.dtype)
            nv = mx.zeros((B, H, cap, values.shape[3]), values.dtype)
            if keep:
                nk[..., :keep, :] = self.keys[..., self.idx - keep : self.idx, :]
                nv[..., :keep, :] = self.values[..., self.idx - keep : self.idx, :]
            self.keys, self.values, self.idx = nk, nv, keep
        self.keys[..., self.idx : self.idx + S, :] = keys
        self.values[..., self.idx : self.idx + S, :] = values
        self.idx += S
        self.offset += S
        return self.keys[..., : self.idx, :], self.values[..., : self.idx, :]

    @property
    def state(self):
        return self.keys, self.values

    def is_trimmable(self):
        return True

    def trim(self, n):
        assert n <= self.slack
        n = min(self.idx, n)
        self.idx -= n
        self.offset -= n
        return n

    def empty(self):
        return self.keys is None


class WindowConvCache(ArraysCache):
    """Conv state that also keeps the full input windows of the last call,
    so the state can be rolled back to any token of that call."""

    keep_windows = True
    windows = None

    def rollback(self, keep: int):
        # Window i has K-1 old rows + L new rows; the state after `keep`
        # new rows is rows [keep, keep + K - 1).
        n = self.cache[0].shape[1]
        self.cache = [mx.contiguous(w[:, keep : keep + n]) for w in self.windows]
        self.windows = None


def make_cache(model, passes: int, rollback_conv: bool):
    set_passes(model, passes)
    window = text_model(model).args.sliding_window_size
    conv_cls = WindowConvCache if rollback_conv else ArraysCache
    return [
        CacheList(
            WindowKVCache(window) if isinstance(l, SlidingDecoderLayer) else KVCache(),
            conv_cls(4),
        )
        for l in text_model(model).layer_visits
    ]


def forward(model, passes, tokens, cache):
    set_passes(model, passes)
    return model(tokens[None], cache=cache)


def prefill(model, passes, tokens, cache, chunk=2048):
    tokens = mx.array(tokens)
    while tokens.size:
        forward(model, passes, tokens[:chunk], cache)
        mx.eval([c.state for c in cache])
        for c in cache:
            if getattr(c[1], "windows", None) is not None:
                c[1].windows = None
        tokens = tokens[chunk:]
        mx.clear_cache()


def plain_greedy(model, prompt, max_tokens, passes=3, cache=None, eos=None):
    """Plain greedy decode with the stock cache. Returns tokens, decode secs."""
    set_passes(model, passes)
    cache = cache if cache is not None else make_prompt_cache(model)
    prefill(model, passes, prompt[:-1], cache)
    y = mx.array(prompt[-1:])

    def step(y):
        return mx.argmax(forward(model, passes, y, cache)[0, -1:], axis=-1)

    y = step(y)
    mx.async_eval(y)
    out = []
    t0 = None
    while len(out) < max_tokens:
        nxt = step(y)
        mx.async_eval(nxt)
        tok = y.item()
        if t0 is None:
            t0 = time.perf_counter()  # first token = prefill end
        out.append(tok)
        if eos is not None and tok in eos:
            break
        y = nxt
    mx.eval(nxt)
    dt = time.perf_counter() - t0
    return out, dt, len(out) - 1


def spec_greedy(
    model, prompt, max_tokens, k, draft_passes, target_passes=3, eos=None
):
    """Greedy self-speculative decode. Output equals greedy decoding of the
    target, up to numerics of a k+1-token forward vs one-token forwards."""
    T = make_cache(model, target_passes, rollback_conv=True)
    D = make_cache(model, draft_passes, rollback_conv=False)
    prefill(model, target_passes, prompt[:-1], T)
    prefill(model, draft_passes, prompt[:-1], D)

    y = prompt[-1]
    d_feed = mx.array([y])
    out, accepts = [], []
    t0 = time.perf_counter()
    n_first = None
    while len(out) < max_tokens:
        drafts, snaps = [], []
        feed = d_feed
        for _ in range(k):
            t = mx.argmax(forward(model, draft_passes, feed, D)[0, -1:], axis=-1)
            mx.async_eval(t)
            snaps.append([list(c[1].cache) for c in D])
            drafts.append(t)
            feed = t
        verify_in = mx.concatenate([mx.array([y])] + drafts)
        g = mx.argmax(forward(model, target_passes, verify_in, T)[0], axis=-1)
        mx.eval(g, drafts)
        d = mx.concatenate(drafts).tolist()
        g = g.tolist()
        a = 0
        while a < k and d[a] == g[a]:
            a += 1
        accepts.append(a)
        new = d[:a] + [g[a]]

        # Target fed y + k drafts; keep y + a drafts.
        for c in T:
            c[1].rollback(a + 1)
            c[0].trim(k - a)
        # Draft call j fed the token before d_{j+1}; keep calls 0..c_keep.
        c_keep = min(a, k - 1)
        for c, s in zip(D, snaps[c_keep]):
            c[1].cache = list(s)
            c[0].trim(k - 1 - c_keep)
        d_feed = mx.array([d[k - 1], g[k]]) if a == k else mx.array([g[a]])
        y = g[a]

        if n_first is None:
            # Match plain_greedy: time from the first emitted token.
            t0 = time.perf_counter()
            n_first = len(new)
        out.extend(new)
        if eos is not None and any(t in eos for t in new):
            i = min(i for i, t in enumerate(out) if t in eos)
            out = out[: i + 1]
            break
    dt = time.perf_counter() - t0
    out = out[:max_tokens]
    return out, dt, len(out) - n_first, accepts
