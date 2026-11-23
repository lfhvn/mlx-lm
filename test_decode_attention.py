"""The batched-decode attention path must match the float32 reference.

Also runs real cache objects past the window, so rotating caches wrap.

Covers B=1 and B=4, left padding (rows with fewer real keys), full attention
and a sliding window, and a relative-position extent shorter than S.
"""

from types import SimpleNamespace

import mlx.core as mx
from mlx_lm.models.inkling2 import layers

Hq, Hkv, D = 24, 8, 128


def case(B, S, window, R, offsets, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((B, Hq, 1, D)).astype(mx.bfloat16)
    k = mx.random.normal((B, Hkv, S, D)).astype(mx.bfloat16)
    v = mx.random.normal((B, Hkv, S, D)).astype(mx.bfloat16)
    pos_bias = mx.random.normal((B, 1, Hq, R)).astype(mx.bfloat16)
    sinks = mx.random.normal((B, 1, Hq)).astype(mx.bfloat16)
    me = SimpleNamespace(scaling=D**-0.5, sliding_window=window)
    end = mx.array(offsets) if B > 1 else offsets[0]
    cache = SimpleNamespace(offset=end)

    new = layers.Attention._attention(me, q, k, v, pos_bias, sinks, cache)
    qp = layers._positions(end - 1, 1)
    kp = layers._key_positions(cache, S, 1, end)
    ref = layers._attention_reference(
        q, k, v, pos_bias, sinks, scale=me.scaling, window=window,
        query_positions=qp, key_positions=kp,
    )
    err = mx.abs(new.astype(mx.float32) - ref.astype(mx.float32)).max().item()
    scale = mx.abs(ref.astype(mx.float32)).max().item()
    ok = new.shape == ref.shape and err < 2e-2 * max(scale, 1.0)
    print(f"B={B} S={S} window={window} R={R} offsets={offsets}: shape={new.shape} "
          f"max_err={err:.2e} (ref max {scale:.2f}) {'OK' if ok else 'FAIL'}")
    return ok


results = [
    case(1, 300, 0, 128, [300]),
    case(4, 300, 0, 128, [300, 300, 300, 300]),
    case(4, 300, 0, 128, [300, 250, 120, 1]),     # left-padded rows
    case(4, 300, 100, 100, [300, 250, 120, 7]),   # sliding window
    case(4, 2048, 0, 1024, [2048, 2000, 1500, 900], seed=1),
]
print("ALL OK" if all(results) else "SOME FAILED")


# Large sinks: the batched path moves each row's sink into its bf16 mask.
def large_sinks():
    mx.random.seed(3)
    B, S = 4, 512
    q = mx.random.normal((B, Hq, 1, D)).astype(mx.bfloat16)
    k = mx.random.normal((B, Hkv, S, D)).astype(mx.bfloat16)
    v = mx.random.normal((B, Hkv, S, D)).astype(mx.bfloat16)
    pos_bias = (4 * mx.random.normal((B, 1, Hq, 256))).astype(mx.bfloat16)
    sinks = (8 * mx.random.normal((B, 1, Hq))).astype(mx.bfloat16)
    me = SimpleNamespace(scaling=D**-0.5, sliding_window=0)
    end = mx.array([512, 400, 300, 50])
    cache = SimpleNamespace(offset=end)
    new = layers.Attention._attention(me, q, k, v, pos_bias, sinks, cache)
    ref = layers._attention_reference(
        q, k, v, pos_bias, sinks, scale=me.scaling, window=0,
        query_positions=layers._positions(end - 1, 1),
        key_positions=layers._key_positions(cache, S, 1, end),
    )
    err = mx.abs(new.astype(mx.float32) - ref.astype(mx.float32)).max().item()
    scale = mx.abs(ref.astype(mx.float32)).max().item()
    ok = err < 2e-2 * max(scale, 1.0)
    print(f"large sinks/bias B=4: max_err={err:.2e} (ref max {scale:.2f}) {'OK' if ok else 'FAIL'}")
    return ok


# Real cache objects, decoded past the window so rotating caches wrap.
from mlx_lm.models.cache import (
    BatchKVCache, BatchRotatingKVCache, KVCache, RotatingKVCache,
)


def cache_run(kind, window, prompt_lens, steps, R, seed=5):
    mx.random.seed(seed)
    B = len(prompt_lens)
    P = max(prompt_lens)
    pads = [P - n for n in prompt_lens]
    if kind == "KVCache":
        cache = KVCache()
    elif kind == "RotatingKVCache":
        cache = RotatingKVCache(max_size=window)
    elif kind == "BatchKVCache":
        cache = BatchKVCache(pads)
    else:
        cache = BatchRotatingKVCache(window, pads)
    me = SimpleNamespace(scaling=D**-0.5, sliding_window=window if "Rotating" in kind else 0)
    hist_k = mx.random.normal((B, Hkv, P, D)).astype(mx.bfloat16)
    hist_v = mx.random.normal((B, Hkv, P, D)).astype(mx.bfloat16)
    cache.update_and_fetch(hist_k, hist_v)
    mx.eval(cache.keys, cache.values)
    worst = 0.0
    ok = True
    for step in range(steps):
        kn = mx.random.normal((B, Hkv, 1, D)).astype(mx.bfloat16)
        vn = mx.random.normal((B, Hkv, 1, D)).astype(mx.bfloat16)
        q = mx.random.normal((B, Hq, 1, D)).astype(mx.bfloat16)
        pos_bias = mx.random.normal((B, 1, Hq, R)).astype(mx.bfloat16)
        sinks = mx.random.normal((B, 1, Hq)).astype(mx.bfloat16)
        keys, values = cache.update_and_fetch(kn, vn)
        new = layers.Attention._attention(me, q, keys, values, pos_bias, sinks, cache)
        hist_k = mx.concatenate([hist_k, kn], axis=2)
        hist_v = mx.concatenate([hist_v, vn], axis=2)
        T = hist_k.shape[2]
        kp = mx.arange(T)[None] - mx.array(pads)[:, None]
        qp = mx.array([[T - 1 - p] for p in pads])
        ref = layers._attention_reference(
            q, hist_k, hist_v, pos_bias, sinks, scale=me.scaling,
            window=me.sliding_window, query_positions=qp, key_positions=kp,
        )
        err = mx.abs(new.astype(mx.float32) - ref.astype(mx.float32)).max().item()
        scale = mx.abs(ref.astype(mx.float32)).max().item()
        worst = max(worst, err / max(scale, 1.0))
        ok = ok and err < 2e-2 * max(scale, 1.0)
    print(f"{kind} window={window} prompts={prompt_lens} steps={steps} R={R}: "
          f"worst rel err={worst:.2e} {'OK' if ok else 'FAIL'}")
    return ok


results += [
    large_sinks(),
    cache_run("KVCache", 0, [40], 30, 32),
    cache_run("BatchKVCache", 0, [40, 25, 3, 40], 30, 32),
    cache_run("RotatingKVCache", 64, [40], 120, 64),
    cache_run("BatchRotatingKVCache", 64, [40, 25, 3, 40], 120, 64),
    cache_run("BatchRotatingKVCache", 64, [100, 70, 10, 90], 80, 64),
    cache_run("BatchRotatingKVCache", 1024, [1000, 600, 50, 1024], 40, 1024),
    cache_run("BatchKVCache", 0, [3000, 2500, 100, 2900], 5, 1024),
]
print("ALL OK (incl. caches)" if all(results) else "SOME FAILED")
