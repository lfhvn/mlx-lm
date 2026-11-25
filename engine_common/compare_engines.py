"""Compare two mlx-lm checkouts on the real model, in two runs:

    <base python> -I compare_engines.py --out base.json      (also writes the reference tokens)
    <new python>  -I compare_engines.py --out new.json
    python3 compare_engines.py --compare base.json new.json

Per run: (1) teacher-forced B=1 decode over the base run's greedy tokens,
recording the top-1 token, top-1/top-2 logits and the log-prob of the
reference token at each step; (2) greedy batch_generate of the 4 prompts at
B=4, 200 tokens.
"""
import argparse, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--out")
ap.add_argument("--compare", nargs=2)
ap.add_argument("--ref", default="/Users/lhm/inkling-orchestra/engine/common/compare_ref_tokens.json")
ap.add_argument("--steps", type=int, default=200)
args = ap.parse_args()

if args.compare:
    a, b = (json.load(open(p)) for p in args.compare)
    agree = total = 0; max_dlp = 0.0; flips = []
    for pa, pb in zip(a["tf"], b["tf"]):
        for i, (sa, sb) in enumerate(zip(pa, pb)):
            total += 1
            agree += sa["top1"] == sb["top1"]
            max_dlp = max(max_dlp, abs(sa["ref_lp"] - sb["ref_lp"]))
            if sa["top1"] != sb["top1"]:
                flips.append(round(sa["margin"], 3))
    print(f"teacher-forced B=1: top-1 agree {agree}/{total}; max |d logprob(ref)| {max_dlp:.4f}; "
          f"base top1-top2 margin at flips: {sorted(flips)}")
    for i, (ta, tb) in enumerate(zip(a["batch"], b["batch"])):
        n = next((k for k, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
        print(f"B=4 greedy prompt {i}: {'identical' if ta == tb else f'first difference at token {n} of {len(ta)}'}")
    sys.exit()

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import batch_generate
from mlx_lm.models.cache import make_prompt_cache
import mlx_lm

print("mlx_lm from", mlx_lm.__file__, flush=True)
model, tok = load("pipenetwork/Inkling-2-Mini-MLX-bf16")
qs = [
    "Write a Python function that returns the longest palindromic substring of s.",
    "Explain in two sentences why the sky is blue.",
    "Given N up to 1e5, count inversions in an array. Describe the algorithm.",
    "What is 17 * 23? Show the steps.",
]
prompts = [tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                   reasoning_effort="low") for q in qs]

if not os.path.exists(args.ref):
    ref = []
    for p in prompts:
        cache = make_prompt_cache(model)
        y = mx.argmax(model(mx.array([p]), cache=cache)[0, -1]).item()
        toks = [y]
        for _ in range(args.steps - 1):
            y = mx.argmax(model(mx.array([[y]]), cache=cache)[0, -1]).item()
            toks.append(y)
        ref.append(toks)
    json.dump(ref, open(args.ref, "w"))
    print("wrote reference tokens", flush=True)
ref = json.load(open(args.ref))

tf = []
for p, toks in zip(prompts, ref):
    cache = make_prompt_cache(model)
    logits = model(mx.array([p]), cache=cache)[0, -1]
    steps = []
    for t in toks:
        lp = logits.astype(mx.float32) - mx.logsumexp(logits.astype(mx.float32))
        top2 = mx.topk(logits.astype(mx.float32), 2)
        top1 = mx.argmax(logits).item()
        hi = sorted(top2.tolist(), reverse=True)
        steps.append({"top1": top1, "margin": hi[0] - hi[1], "ref_lp": lp[t].item()})
        logits = model(mx.array([[t]]), cache=cache)[0, -1]
    tf.append(steps)

res = batch_generate(model, tok, prompts, max_tokens=args.steps, return_token_ids=True)
json.dump({"tf": tf, "batch": res.token_ids}, open(args.out, "w"))
print("wrote", args.out)
