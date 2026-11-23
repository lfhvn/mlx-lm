"""Inkling 2 DSpark draft experiments (run under ../common/gpulock.py).

  offline: target greedy generations, then the mean accepted length of a draft
           block at every anchor position (teacher forced).
  spec:    greedy target decode vs speculative decode, B=1, back to back:
           tokens must match; prints tok/s and acceptance.
"""

import argparse, json, statistics, time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.inkling2.dspark import load_draft
from mlx_lm.models.inkling2.speculative import (
    _forward,
    greedy_generate,
    speculative_generate,
)

HERE = Path(__file__).resolve().parent
QUESTIONS = [
    "Write a Python function that returns the longest palindromic substring of s.",
    "Explain in two sentences why the sky is blue.",
    "Given N up to 1e5, count inversions in an array. Describe the algorithm.",
    "What is 17 * 23? Show the steps.",
    "Write a short story (about 150 words) about a lighthouse keeper.",
    "Refactor this into idiomatic Python:\nresult = []\nfor i in range(len(xs)):\n    if xs[i] % 2 == 0:\n        result.append(xs[i] * xs[i])",
]

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["offline", "spec"])
ap.add_argument("--model", default="pipenetwork/Inkling-2-Mini-MLX-bf16")
ap.add_argument("--weights", default=str(HERE.parent / "weights"))
ap.add_argument("--max-tokens", type=int, default=256)
ap.add_argument("--n", type=int, default=len(QUESTIONS))
ap.add_argument("--effort", default="low")
ap.add_argument("--out", default=None)
ap.add_argument("--num-draft", default="7", help="spec: comma list, e.g. 3,7")
ap.add_argument("--variants", default="ref", help="offline: ref,shift,nomarkov")
args = ap.parse_args()

model, tok = load(args.model)
draft = load_draft(model, args.weights)
eos = set(tok.eos_token_ids) if hasattr(tok, "eos_token_ids") else {tok.eos_token_id}
prompts = [
    tok.apply_chat_template(
        [{"role": "user", "content": q}],
        add_generation_prompt=True,
        reasoning_effort=args.effort,
    )
    for q in QUESTIONS[: args.n]
]
results = []


def offline(prompt, gen=None, variant="ref"):
    if gen is None:
        gen, _ = greedy_generate(model, prompt, args.max_tokens, eos)
    seq = prompt + gen
    ids = list(draft.args.target_layer_ids)
    if variant == "shift":
        ids = [i + 1 for i in ids[:-1]] + ids[-1:]
    draft.no_markov = variant == "nomarkov"
    _, feats = _forward(model, mx.array(seq)[None], None, ids)
    mx.eval(feats)
    draft.reset()
    draft.store_context(feats, 0)
    mx.eval([c.keys for c in draft.ctx])
    n = draft.args.block_size - 1
    acc = []
    for p in range(len(prompt), len(seq) - n):
        for c in draft.ctx:
            c.offset = p
        d = draft.draft(mx.array([seq[p]]), p, [f[:, p - 1 : p] for f in feats])
        d = d[0].tolist()
        a = 0
        while a < n and d[a] == seq[p + 1 + a]:
            a += 1
        acc.append(a)
    return gen, acc


for i, prompt in enumerate(prompts):
    if args.mode == "offline":
      gen = None
      for v in args.variants.split(","):
        gen, acc = offline(prompt, gen, v)
        m = statistics.mean(acc)
        print(f"[{i}] {v} gen={len(gen)} anchors={len(acc)} mean_accepted={m:.2f} "
              f"tokens/round={m + 1:.2f} hist={[acc.count(k) for k in range(8)]}", flush=True)
        results.append({"q": QUESTIONS[i], "variant": v, "accepted": acc, "gen": gen})
    else:
        greedy_generate(model, prompt, 16, eos)  # warm up
        speculative_generate(model, draft, prompt, 16, eos)
        base, t_base = greedy_generate(model, prompt, args.max_tokens, eos)
        for nd in [int(x) for x in args.num_draft.split(",")]:
            stats = {}
            spec, t_spec = speculative_generate(
                model, draft, prompt, args.max_tokens, eos, stats, num_draft=nd)
            prefix = next((k for k, (a, b) in enumerate(zip(base, spec)) if a != b),
                          min(len(base), len(spec)))
            acc = stats["accepted"]
            r = {
                "q": QUESTIONS[i], "num_draft": nd, "identical": base == spec,
                "first_diff": prefix, "n_base": len(base), "n_spec": len(spec),
                "base_tok_s": (len(base) - 1) / t_base,
                "spec_tok_s": (len(spec) - 1) / t_spec,
                "mean_accepted": statistics.mean(acc), "rounds": len(acc),
                "accepted": acc, "base": base, "spec": spec,
            }
            print(f"[{i}] nd={nd} identical={r['identical']} first_diff={prefix}/{len(base)} "
                  f"base {r['base_tok_s']:.1f} tok/s  spec {r['spec_tok_s']:.1f} tok/s "
                  f"(x{r['spec_tok_s'] / r['base_tok_s']:.2f})  "
                  f"mean_accepted={r['mean_accepted']:.2f}", flush=True)
            if not r["identical"]:
                print("   base:", repr(tok.decode(base[max(0, prefix - 5) : prefix + 10])))
                print("   spec:", repr(tok.decode(spec[max(0, prefix - 5) : prefix + 10])))
            results.append(r)

if args.mode == "offline":
  for v in args.variants.split(","):
    allacc = [a for r in results if r["variant"] == v for a in r["accepted"]]
    print(f"ALL {v}: anchors={len(allacc)} mean_accepted={statistics.mean(allacc):.3f} "
          f"tokens/round={statistics.mean(allacc) + 1:.3f}")
else:
  for nd in [int(x) for x in args.num_draft.split(",")]:
    rs = [r for r in results if r["num_draft"] == nd]
    base_t = sum(r["n_base"] - 1 for r in rs) / sum((r["n_base"] - 1) / r["base_tok_s"] for r in rs)
    spec_t = sum(r["n_spec"] - 1 for r in rs) / sum((r["n_spec"] - 1) / r["spec_tok_s"] for r in rs)
    acc = [a for r in rs for a in r["accepted"]]
    print(f"ALL nd={nd}: identical {sum(r['identical'] for r in rs)}/{len(rs)}  "
          f"base {base_t:.1f} tok/s  spec {spec_t:.1f} tok/s (x{spec_t / base_t:.2f})  "
          f"mean_accepted {statistics.mean(acc):.2f}")
if args.out:
    Path(args.out).write_text(json.dumps(results))
