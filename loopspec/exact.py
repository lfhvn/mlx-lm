"""(c) Exactness: speculative greedy vs plain 3-pass greedy (stock cache,
one token per step). At each first divergence, report plain's top-1 minus
top-2 logit margin there."""
import sys, json
sys.path.insert(0, "/Users/lhm/inkling-orchestra/engine/loop-spec")
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from loopspec.spec import spec_greedy, prefill, forward, set_passes
from loopspec.shared import SharedSpec
from loopspec.prompts import chat_prompts

N = 128
model, tok = load("/Users/lhm/models/TML-d24-varloop-mlx")


def plain_with_margin(p):
    set_passes(model, 3)
    c = make_prompt_cache(model)
    prefill(model, 3, p[:-1], c)
    y, out, margins = p[-1], [], []
    for _ in range(N):
        l = forward(model, 3, mx.array([y]), c)[0, -1]
        top = mx.topk(l, 2)
        y = mx.argmax(l).item()
        t = sorted(top.tolist(), reverse=True)
        out.append(y); margins.append(t[0] - t[1])
    return out, margins


res = []
for pi, p in enumerate(chat_prompts(tok)):
    ref, m = plain_with_margin(p)
    row = {"prompt": pi, "n": N, "plain_margins_lt_0.1": sum(x < 0.1 for x in m),
           "plain_margins_lt_0.25": sum(x < 0.25 for x in m)}
    for name, fn in [("spec dp1 k3", lambda: spec_greedy(model, p, N, 3, 1)),
                     ("shared dp1 k3", lambda: SharedSpec(model, 1).generate(p, N, 3)),
                     ("shared dp2 k3", lambda: SharedSpec(model, 2).generate(p, N, 3))]:
        o = fn()[0]
        i = next((i for i, (a, b) in enumerate(zip(o, ref)) if a != b), None)
        row[name] = {"first_diff": i, "margin": None if i is None else round(m[i], 4)}
    res.append(row)
    print(row, flush=True)
json.dump(res, open("/Users/lhm/inkling-orchestra/engine/loop-spec/loopspec/out/exact.json", "w"), indent=1)
