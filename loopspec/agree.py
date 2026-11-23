"""(a) Teacher-forced next-token agreement of 1/2-pass (and 3-pass) with
3-pass greedy text, plus the speculative acceptance it predicts."""
import sys, json
sys.path.insert(0, "/Users/lhm/inkling-orchestra/engine/loop-spec")
import mlx.core as mx
from mlx_lm import load
from loopspec.spec import plain_greedy, make_cache, prefill, forward
from loopspec.prompts import chat_prompts

N = int(sys.argv[1]) if len(sys.argv) > 1 else 256
model, tok = load("/Users/lhm/models/TML-d24-varloop-mlx")
rows = []
for pi, p in enumerate(chat_prompts(tok)):
    ref, dt, n = plain_greedy(model, p, N)
    seq = p + ref
    row = {"prompt": pi, "plen": len(p), "n": len(ref), "match": {}}
    for passes in (1, 2, 3, 5):
        c = make_cache(model, passes, True)
        prefill(model, passes, seq[: len(p) - 1], c)
        l = forward(model, passes, mx.array(seq[len(p) - 1 : -1]), c)[0]
        pred = mx.argmax(l, axis=-1).tolist()
        row["match"][passes] = [int(a == b) for a, b in zip(pred, ref)]
        mx.clear_cache()
    rows.append(row)
    print(pi, {k: sum(v) / len(v) for k, v in row["match"].items()}, flush=True)
json.dump(rows, open("/Users/lhm/inkling-orchestra/engine/loop-spec/loopspec/out/agree.json", "w"))
