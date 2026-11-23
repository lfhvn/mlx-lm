import sys
sys.path.insert(0, "/Users/lhm/inkling-orchestra/engine/loop-spec")
import mlx.core as mx
from mlx_lm import load
from loopspec.spec import plain_greedy, make_cache, prefill, forward
from loopspec.shared import SharedSpec
from loopspec.prompts import chat_prompts

model, tok = load("/Users/lhm/models/TML-d24-varloop-mlx")
p = chat_prompts(tok)[0]
for dp in (1, 2):
    s = SharedSpec(model, dp)
    s.prefill(p[:-1])
    pre, early, dt = s._shared(mx.array(p[-1:]))
    a = s._target(pre, early, dt)[0, -1]
    d = s._draft_coda(*early, dt)[0, -1]
    c = make_cache(model, 3, True); prefill(model, 3, p[:-1], c)
    b = forward(model, 3, mx.array(p[-1:]), c)[0, -1]
    c = make_cache(model, dp, True); prefill(model, dp, p[:-1], c)
    e = forward(model, dp, mx.array(p[-1:]), c)[0, -1]
    print(f"dp={dp}: target max diff {mx.abs(a-b).max().item()}, draft max diff {mx.abs(d-e).max().item()}")
ref, _, _ = plain_greedy(model, p, 64)
for dp in (1, 2):
    o, dt, n, acc = SharedSpec(model, dp).generate(p, 64, 3)
    first = next((i for i, (x, y) in enumerate(zip(o, ref)) if x != y), None)
    print(f"shared dp={dp}: {n/dt:.2f} tok/s acc {sum(acc)/len(acc):.2f} first_diff {first}")
    print(repr(tok.decode(o[:80])))
