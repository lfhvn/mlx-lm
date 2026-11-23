"""(b)-(d) Speculative vs plain 3-pass greedy at B=1, run back to back.

    bench.py --ctx short|4k --dp 1 --ks 2,3,4,5,6 --n 128 --prompts 0,1,2
"""
import argparse, json, sys, time
sys.path.insert(0, "/Users/lhm/inkling-orchestra/engine/loop-spec")
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from loopspec.spec import plain_greedy, spec_greedy, make_cache, prefill, forward
from loopspec.shared import SharedSpec
from loopspec.prompts import chat_prompts, long_prompt

ap = argparse.ArgumentParser()
ap.add_argument("--ctx", default="short")
ap.add_argument("--dp", default="1")
ap.add_argument("--ks", default="2,3,4,5,6")
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--prompts", default="0,1,2")
ap.add_argument("--micro", action="store_true")
ap.add_argument("--tag", default="")
ap.add_argument("--mode", default="spec")
args = ap.parse_args()

model, tok = load("/Users/lhm/models/TML-d24-varloop-mlx")
eos = {tok.eos_token_id} if tok.eos_token_id is not None else None
if args.ctx == "short":
    prompts = [chat_prompts(tok)[int(i)] for i in args.prompts.split(",")]
else:
    prompts = [long_prompt(tok)]


def micro(ctx):
    """ms per forward at each pass count and chunk length L."""
    res = {}
    for passes in (1, 2, 3):
        c = make_cache(model, passes, True)
        prefill(model, passes, ctx, c)
        for L in (1, 2, 4, 7):
            ts = []
            for _ in range(8):
                mx.synchronize(); t = time.perf_counter()
                mx.eval(forward(model, passes, mx.array(ctx[-L:]), c))
                mx.synchronize(); ts.append(time.perf_counter() - t)
                for cc in c:
                    cc[1].rollback(0) if cc[1].windows is not None else None
                    cc[0].trim(L)
            res[f"p{passes}_L{L}"] = sorted(ts)[3] * 1e3
            print(f"passes={passes} L={L}: {res[f'p{passes}_L{L}']:.1f} ms", flush=True)
    return res


out = {"ctx": args.ctx, "runs": []}
if args.micro:
    out["micro"] = micro(prompts[0])
for pi, p in enumerate(prompts):
    ref, dt, n = plain_greedy(model, p, args.n, eos=eos)
    base = n / dt
    print(f"prompt {pi} len {len(p)}: plain {base:.2f} tok/s ({len(ref)} tok)", flush=True)
    out["runs"].append({"prompt": pi, "mode": "plain", "tps": base, "n": len(ref)})
    for mode, dp in [(m, int(x)) for x in args.dp.split(",") for m in args.mode.split(",")]:
        for k in [int(x) for x in args.ks.split(",")]:
            if mode == "shared":
                o, dt, n, acc = SharedSpec(model, dp).generate(p, args.n, k, eos=eos)
            else:
                o, dt, n, acc = spec_greedy(model, p, args.n, k, dp, eos=eos)
            first = next((i for i, (a, b) in enumerate(zip(o, ref)) if a != b), None)
            r = {"prompt": pi, "mode": mode, "dp": dp, "k": k, "tps": n / dt,
                 "mean_accept": sum(acc) / len(acc), "rounds": len(acc),
                 "tok_per_round": len(o) / len(acc), "exact": o == ref,
                 "first_diff": first, "n": len(o)}
            out["runs"].append(r)
            print(f"  {mode} dp={dp} k={k}: {r['tps']:.2f} tok/s ({r['tps']/base:.2f}x) "
                  f"acc {r['mean_accept']:.2f}/{k} tok/round {r['tok_per_round']:.2f} "
                  f"exact={r['exact']} first_diff={first}", flush=True)
            mx.clear_cache()
json.dump(out, open(f"/Users/lhm/inkling-orchestra/engine/loop-spec/loopspec/out/bench_{args.ctx}_{args.mode.replace(',', '+')}_dp{args.dp.replace(',', '')}{args.tag}.json", "w"), indent=1)
