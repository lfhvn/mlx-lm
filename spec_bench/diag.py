"""Why spec output differs from greedy: compare the target's argmax at each
position when decoding 1 token per forward (greedy) against
(a) one prefill forward over the whole sequence (stock caches) and
(b) 8-token verify chunks of the true sequence (spec caches + rollback).
Prints the top-1/top-2 logit gap of greedy at every disagreement."""
import json, sys
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.inkling2.speculative import make_spec_cache, prefill, _forward, _rollback

model, tok = load("pipenetwork/Inkling-2-Mini-MLX-bf16")
res = json.load(open(sys.argv[1]))
qs = [r["q"] for r in res]
for r in res:
    if r["identical"]:
        continue
    prompt = tok.apply_chat_template([{"role": "user", "content": r["q"]}],
                                     add_generation_prompt=True, reasoning_effort="low")
    base = r["base"]
    # Greedy, recording the top-2 gap at each step.
    cache = model.make_cache()
    logits, _ = prefill(model, prompt, cache)
    gaps, lg = [], logits
    for i, t in enumerate(base):
        top = mx.topk(lg[0], 2)
        mx.eval(top)
        s = sorted(top.tolist(), reverse=True)
        gaps.append(s[0] - s[1])
        assert mx.argmax(lg[0]).item() == t
        lg = model(mx.array([[t]]), cache=cache)[:, -1]
    seq = prompt + base
    # (a) one forward over everything.
    full = model(mx.array([seq[:-1]]))[0, len(prompt) - 1 :]
    a = mx.argmax(full, axis=-1).tolist()
    # (b) 8-token chunks with spec caches, keeping all 8 (true tokens).
    cache = make_spec_cache(model)
    logits, _ = prefill(model, prompt, cache)
    b = [mx.argmax(logits[0]).item()]
    for s in range(0, len(base) - 1, 8):
        chunk = base[s : s + 8]
        lg, _ = _forward(model, mx.array([chunk]), cache, [5, 48])
        b += mx.argmax(lg[0], axis=-1).tolist()
    b = b[: len(base)]
    da = [(i, round(gaps[i], 3)) for i in range(len(base)) if a[i] != base[i]]
    db = [(i, round(gaps[i], 3)) for i in range(len(base)) if b[i] != base[i]]
    print(f"{r['q'][:40]!r}: spec first diff {r['first_diff']} (gap {gaps[r['first_diff']]:.3f})")
    print("   full-forward disagreements (pos, greedy gap):", da[:10])
    print("   8-chunk disagreements (pos, greedy gap):", db[:10])
    print("   median greedy gap:", sorted(gaps)[len(gaps) // 2])
