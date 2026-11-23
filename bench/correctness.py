"""Dump logits and greedy tokens of the real model, for a base/fuse diff.
usage: python correctness.py out.npz   (then: correctness.py --cmp a.npz b.npz)"""
import json, sys
import numpy as np

if sys.argv[1] == "--cmp":
    a, b = np.load(sys.argv[2]), np.load(sys.argv[3])
    for k in sorted(a.files):
        if k.startswith("logits"):
            d = np.abs(a[k] - b[k])
            print(f"{k}: max abs diff {d.max():.4g} (logit range {np.abs(a[k]).max():.3g}), "
                  f"argmax same {np.mean(a[k].argmax(-1) == b[k].argmax(-1)):.3f}")
        else:
            same = np.array_equal(a[k], b[k])
            first = next((i for i, (x, y) in enumerate(zip(a[k].ravel(), b[k].ravel())) if x != y), None)
            print(f"{k}: tokens identical={same}" + ("" if same else f", first diff at {first}"))
    sys.exit()

import mlx.core as mx
import mlx_lm
from mlx_lm import load
from mlx_lm.generate import batch_generate, generate_step
from mlx_lm.models.cache import make_prompt_cache

print("mlx_lm from", mlx_lm.__file__)
model, tok = load("pipenetwork/Inkling-2-Mini-MLX-bf16")
qs = [
    "Write a Python function that returns the longest palindromic substring of s.",
    "Explain in two sentences why the sky is blue.",
    "Given N up to 1e5, count inversions in an array. Describe the algorithm.",
    "What is 17 * 23? Show the steps.",
]
prompts = [tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                   reasoning_effort="low") for q in qs]
N = 200
out = {}
for i, p in enumerate(prompts[:3]):
    # B=1 greedy decode with a plain cache; keep the logits of the first 32 steps.
    cache = make_prompt_cache(model)
    toks, logits = [], []
    for n, (t, lp) in zip(range(N), generate_step(mx.array(p), model, prompt_cache=cache)):
        toks.append(t)
        if n < 32:
            logits.append(np.array(lp.astype(mx.float32)))
    out[f"tokens_b1_{i}"] = np.array(toks)
    out[f"logits_b1_{i}"] = np.stack(logits)
# BatchGenerator path, B=4 with different prompts.
res = batch_generate(model, tok, prompts, max_tokens=N)
for i, t in enumerate(res.texts):
    out[f"tokens_b4_{i}"] = np.array(tok.encode(t))
np.savez(sys.argv[1], **out)
print("saved", sys.argv[1])
