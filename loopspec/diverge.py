"""Is the divergence numeric? Compare target logits at the divergence point
from (a) one-token decode steps and (b) multi-token forwards."""
import sys
sys.path.insert(0, "/Users/lhm/inkling-orchestra/engine/loop-spec")
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from loopspec.spec import plain_greedy, spec_greedy, make_cache, prefill, forward, set_passes
from loopspec.prompts import chat_prompts

model, tok = load("/Users/lhm/models/TML-d24-varloop-mlx")
p = chat_prompts(tok)[0]
N = 64
ref, _, _ = plain_greedy(model, p, N)
out, _, _, acc = spec_greedy(model, p, N, 3, 1)
i = next(i for i, (a, b) in enumerate(zip(out, ref)) if a != b)
print("diverge at", i, "plain", ref[i], repr(tok.decode([ref[i]])), "spec", out[i], repr(tok.decode([out[i]])))
seq = p + ref[:i]  # context to predict token i

def top2(l):
    l = l.astype(mx.float32)
    o = mx.argsort(-l)[:2].tolist()
    return [(t, round(l[t].item(), 3)) for t in o]

# (a) decode one token at a time
c = make_prompt_cache(model); set_passes(model, 3)
prefill(model, 3, seq[:len(p) - 1], c)
for t in seq[len(p) - 1:]:
    l = forward(model, 3, mx.array([t]), c)[0, -1]
print("one-token steps :", top2(l))
# (b) prefill all but last, then chunks of 4 ending at the target position
for L in (2, 4, 8, len(seq)):
    c = make_cache(model, 3, True)
    prefill(model, 3, seq[: len(seq) - L], c)
    l = forward(model, 3, mx.array(seq[len(seq) - L:]), c)[0, -1]
    print(f"{L:3d}-token forward:", top2(l))
