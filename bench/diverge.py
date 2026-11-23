"""Prompt 0 of correctness.py, greedy 200 tokens at B=1, twice in one process.
Saves the logprobs of every step of both runs, to find where tokens split."""
import sys
import numpy as np
import mlx.core as mx
import mlx_lm
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache

print("mlx_lm from", mlx_lm.__file__)
model, tok = load("pipenetwork/Inkling-2-Mini-MLX-bf16")
q = "Write a Python function that returns the longest palindromic substring of s."
p = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                            reasoning_effort="low")
out = {}
for run in range(int(sys.argv[2]) if len(sys.argv) > 2 else 2):
    cache = make_prompt_cache(model)
    toks, lps = [], []
    for n, (t, lp) in zip(range(int(sys.argv[3]) if len(sys.argv) > 3 else 200), generate_step(mx.array(p), model, prompt_cache=cache)):
        toks.append(t); lps.append(np.array(lp.astype(mx.float32)))
    out[f"tokens{run}"] = np.array(toks); out[f"lp{run}"] = np.stack(lps)
    print("run", run, "done")
np.savez(sys.argv[1], **out)
