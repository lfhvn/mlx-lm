"""Tables for RESULTS.md from loopspec/out/bench_*.json."""
import json, glob, collections
rows = collections.defaultdict(list)
for f in sorted(glob.glob("loopspec/out/bench_*.json")):
    d = json.load(open(f))
    base = {}
    for r in d["runs"]:
        if r["mode"] == "plain":
            base = r
            continue
        key = (d["ctx"], r["mode"], r["dp"], r["k"])
        rows[key].append((r["tps"] / base["tps"], r["tps"], base["tps"], r["mean_accept"], r["tok_per_round"]))
print("| ctx | variant | draft passes | k | runs | speedup mean (min-max) | spec tok/s | plain tok/s | accepted/k | tok/round |")
print("|---|---|---|---|---|---|---|---|---|---|")
for key in sorted(rows):
    v = rows[key]; n = len(v)
    s = [x[0] for x in v]
    m = lambda i: sum(x[i] for x in v) / n
    print(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {n} | {m(0):.2f}x ({min(s):.2f}-{max(s):.2f}) | {m(1):.1f} | {m(2):.1f} | {m(3):.2f}/{key[3]} | {m(4):.2f} |")
