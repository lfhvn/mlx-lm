"""Rollback exactness: verify chunks whose rejected tail is the true text vs
random garbage must leave bit-identical target state."""
import sys, random
sys.path.insert(0, "/Users/lhm/inkling-orchestra/engine/loop-spec")
import mlx.core as mx
from mlx_lm import load
from loopspec.spec import plain_greedy, make_cache, prefill, forward
from loopspec.prompts import chat_prompts

model, tok = load("/Users/lhm/models/TML-d24-varloop-mlx")
p = chat_prompts(tok)[2]
ref, _, _ = plain_greedy(model, p, 120)
seq = p + ref
k = 4
rng = random.Random(0)
pattern = [rng.randint(0, k) for _ in range(40)]


def run(garbage: bool, ctx_pad=0):
    T = make_cache(model, 3, True)
    prefill(model, 3, seq[: len(p) - 1], T)
    pos = len(p) - 1
    logits = []
    for a in pattern:
        chunk = seq[pos : pos + k + 1]
        if garbage:
            chunk = chunk[: a + 1] + [rng.randint(100, 150000) for _ in range(k - a)]
        l = forward(model, 3, mx.array(chunk), T)[0]
        mx.eval(l)
        logits.append(l[: a + 1])
        for c in T:
            c[1].rollback(a + 1)
            c[0].trim(k - a)
        pos += a + 1
        if pos + k + 1 > len(seq):
            break
    return mx.concatenate(logits)


A = run(False)
B = run(True)
d = mx.abs(A - B).max().item()
print(f"positions {A.shape[0]}, max |logit diff| true-tail vs garbage-tail: {d}")
