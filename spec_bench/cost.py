"""ms per target forward of L tokens (B=1) at a short context, and ms per
draft block. Run under ../common/gpulock.py."""
import time, sys
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.inkling2.dspark import load_draft
from mlx_lm.models.inkling2.speculative import make_spec_cache, prefill, _forward, _rollback

model, tok = load("pipenetwork/Inkling-2-Mini-MLX-bf16")
draft = load_draft(model, sys.argv[1] if len(sys.argv) > 1 else "weights")
prompt = tok.encode("def fib(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n" * 8)
ids = draft.args.target_layer_ids

def bench(fn, n=20):
    for _ in range(3): fn()
    mx.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    mx.synchronize(); return (time.perf_counter() - t) / n * 1e3

for kind in ["plain", "spec"]:
    cache = model.make_cache() if kind == "plain" else make_spec_cache(model)
    prefill(model, prompt, cache)
    for L in [1, 2, 4, 8]:
        if kind == "plain" and L > 1: continue
        toks = mx.array([prompt[-L:]])
        def step():
            if kind == "plain":
                out = model(toks, cache=cache)
                mx.eval(out)
                for c in cache: c[0].trim(L)
            else:
                out, f = _forward(model, toks, cache, ids)
                mx.eval(out, f)
                _rollback(cache, 0, L)
        print(f"{kind:5s} target L={L}: {bench(step):6.1f} ms", flush=True)

cache = make_spec_cache(model)
_, feats = prefill(model, prompt, cache, ids)
draft.reset(); draft.store_context(feats, 0)
y = mx.array([prompt[-1]]); last = [f[:, -1:] for f in feats]
print(f"draft block: {bench(lambda: mx.eval(draft.draft(y, len(prompt), last))):6.1f} ms")
print(f"draft backbone only: {bench(lambda: mx.eval(draft.backbone(y, len(prompt)))):6.1f} ms")
