import sys, time
sys.path.insert(0, "/Users/lhm/inkling-orchestra/engine/loop-spec")
import mlx.core as mx
from mlx_lm import load
from loopspec.spec import plain_greedy, spec_greedy, make_cache
from loopspec.prompts import chat_prompts

t = time.time()
model, tok = load("/Users/lhm/models/TML-d24-varloop-mlx")
print(f"load {time.time()-t:.0f}s", flush=True)
p = chat_prompts(tok)[0]
print("prompt len", len(p))
N = 64
ref, dt, n = plain_greedy(model, p, N)
print(f"plain stock: {n/dt:.1f} tok/s")
print(repr(tok.decode(ref)))
ref2, dt, n = plain_greedy(model, p, N, cache=make_cache(model, 3, True))
print(f"plain window cache: {n/dt:.1f} tok/s, same={ref2==ref}")
for dp in (1, 2):
    for k in (3,):
        out, dt, n, acc = spec_greedy(model, p, N, k, dp)
        first = next((i for i, (a, b) in enumerate(zip(out, ref)) if a != b), None)
        print(f"dp={dp} k={k}: {n/dt:.1f} tok/s, mean acc {sum(acc)/len(acc):.2f}, exact={out==ref}, first diff {first}", flush=True)
