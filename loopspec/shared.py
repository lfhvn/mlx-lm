"""Self-speculative decoding that shares the early loop passes.

Loop pass j only reads the prelude output and pass j-1, never later passes.
So the prelude and passes 1..dp of the 3-pass target are the same computation
as the dp-pass draft. Draft and target share those caches and activations;
the draft adds its own coda (layers 18-23), and verify runs only passes
dp+1..3 and the target coda on the k+1 tokens.
"""

import time

import mlx.core as mx

from mlx_lm.models.cache import ArraysCache, CacheList, KVCache
from mlx_lm.models.inkling2.layers import SlidingDecoderLayer, unweighted_rms_norm
from loopspec.spec import WindowConvCache, WindowKVCache, text_model


def _caches(layers, window, conv_cls):
    return [
        CacheList(
            WindowKVCache(window) if isinstance(l, SlidingDecoderLayer) else KVCache(),
            conv_cls(4),
        )
        for l in layers
    ]


class SharedSpec:
    def __init__(self, model, draft_passes, target_passes=3):
        self.model, self.tm = model, text_model(model)
        a = self.tm.args
        self.core = a.loop_core
        self.dp, self.tp = draft_passes, target_passes
        L = self.tm.layers
        w = a.sliding_window_size
        pre, core, coda = L[: self.core.start], L[self.core.start : self.core.stop], L[self.core.stop :]
        self.shared = _caches(pre + core * draft_passes, w, ArraysCache)
        self.tgt = _caches(core * (target_passes - draft_passes) + coda, w, WindowConvCache)
        self.dcoda = _caches(coda, w, ArraysCache)

    def _run(self, layers, h, bs, caches, bank=None):
        for i, (layer, c) in enumerate(zip(layers, caches)):
            msg = None if bank is None else self.tm.message_input(bank, i)
            h, bs = layer(h, bs, cache=c, messages=msg)
        return h, bs

    def _passes(self, pre, prev, start, stop, caches):
        h, bs = prev
        n = len(self.core)
        core = self.tm.layers[self.core.start : self.core.stop]
        for j in range(start, stop):
            bank = None
            if j > 0:
                bank = (bs[pre[1].shape[0] + 1 :], unweighted_rms_norm(h.astype(mx.float32)))
                h, bs = pre
            h, bs = self._run(core, h, bs, caches[(j - start) * n : (j - start + 1) * n], bank)
        return h, bs

    def _head(self, h, bs, dtype):
        normed = unweighted_rms_norm(h.astype(mx.float32))
        h = self.tm.final_input(bs, normed)[..., 0, :]
        h = self.tm.norm(h).astype(dtype)
        logits = self.model.lm_head(h).astype(mx.float32)
        t = self.model.args.text
        logits = mx.tanh(logits / t.final_logit_softcapping) * t.final_logit_softcapping
        return logits[..., : t.unpadded_vocab_size]

    def _shared(self, tokens):
        """Prelude + passes 1..dp on shared caches. Returns (pre, early)."""
        x = self.tm.embed(tokens[None])
        npre = self.core.start
        pre = self._run(self.tm.layers[:npre], x, None, self.shared[:npre])
        early = self._passes(pre, pre, 0, self.dp, self.shared[npre:])
        return pre, early, x.dtype

    def _target(self, pre, early, dtype):
        h, bs = self._passes(pre, early, self.dp, self.tp, self.tgt)
        h, bs = self._run(self.tm.layers[self.core.stop :], h, bs, self.tgt[-len(self.dcoda) :])
        return self._head(h, bs, dtype)

    def _draft_coda(self, h, bs, dtype):
        h, bs = self._run(self.tm.layers[self.core.stop :], h, bs, self.dcoda)
        return self._head(h, bs, dtype)

    def prefill(self, tokens, chunk=2048):
        tokens = mx.array(tokens)
        while tokens.size:
            pre, early, dt = self._shared(tokens[:chunk])
            self._draft_coda(*early, dt)
            self._target(pre, early, dt)
            mx.eval([c.state for c in self.shared + self.tgt + self.dcoda])
            for c in self.tgt:
                c[1].windows = None
            tokens = tokens[chunk:]
            mx.clear_cache()

    def generate(self, prompt, max_tokens, k, eos=None):
        self.prefill(prompt[:-1])
        y = prompt[-1]
        pend = None  # (h, bs) of a token the draft coda has not seen yet
        out, accepts = [], []
        t0, n_first = time.perf_counter(), None
        cat = lambda xs, ax: mx.concatenate(xs, axis=ax)
        while len(out) < max_tokens:
            feed = mx.array([y])
            pres, earlies, drafts, s_snap, d_snap = [], [], [], [], []
            for j in range(k):
                pre, early, dt = self._shared(feed)
                s_snap.append([list(c[1].cache) for c in self.shared])
                pres.append(pre); earlies.append(early)
                h, bs = early
                if j == 0 and pend is not None:
                    h, bs = cat([pend[0], h], 1), cat([pend[1], bs], 2)
                logits = self._draft_coda(h, bs, dt)
                d_snap.append([list(c[1].cache) for c in self.dcoda])
                t = mx.argmax(logits[0, -1:], axis=-1)
                mx.async_eval(t)
                drafts.append(t)
                feed = t
            # Verify: shared part of d_k, then the rest of the target for all.
            pre, early, dt = self._shared(feed)
            s_snap.append([list(c[1].cache) for c in self.shared])
            pres.append(pre); earlies.append(early)
            P = (cat([p[0] for p in pres], 1), cat([p[1] for p in pres], 2))
            E = (cat([e[0] for e in earlies], 1), cat([e[1] for e in earlies], 2))
            g = mx.argmax(self._target(P, E, dt)[0], axis=-1)
            mx.eval(g, drafts)
            d = mx.concatenate(drafts).tolist()
            g = g.tolist()
            a = 0
            while a < k and d[a] == g[a]:
                a += 1
            accepts.append(a)
            new = d[:a] + [g[a]]

            # Shared: k+1 one-token calls (y, d_1..d_k); keep a+1.
            for c, s in zip(self.shared, s_snap[a]):
                c[1].cache = list(s)
                c[0].trim(k - a)
            for c in self.tgt:
                c[1].rollback(a + 1)
                c[0].trim(k - a)
            # Draft coda saw y, d_1..d_{k-1}; keep through d_a.
            ck = min(a, k - 1)
            for c, s in zip(self.dcoda, d_snap[ck]):
                c[1].cache = list(s)
                c[0].trim(k - 1 - ck)
            pend = (earlies[k][0], earlies[k][1]) if a == k else None
            y = g[a]

            if n_first is None:
                t0, n_first = time.perf_counter(), len(new)
            out.extend(new)
            if eos is not None and any(t in eos for t in new):
                out = out[: min(i for i, t in enumerate(out) if t in eos) + 1]
                break
        dt = time.perf_counter() - t0
        out = out[:max_tokens]
        return out, dt, len(out) - n_first, accepts
