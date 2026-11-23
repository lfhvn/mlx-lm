"""Op count and graph-build time of one decode step on a small random model
with the real Inkling 2 Mini layer structure (24 layers, loop 6-18 x3).
No real weights, so it needs no GPU lock. Run with either checkout's python."""
import collections, json, re, sys, time
import mlx.core as mx
import mlx.nn as nn
import mlx_lm
from mlx_lm.models import inkling2
from mlx_lm.models.cache import make_prompt_cache

SNAP = "/Users/lhm/.cache/huggingface/hub/models--pipenetwork--Inkling-2-Mini-MLX-bf16/snapshots/795acb02dcc72fc3714eb723e2bb4dff76a7a94e/config.json"
cfg = json.load(open(SNAP))
t = cfg["text_config"]
t.update(hidden_size=256, vocab_size=512, unpadded_vocab_size=500, n_routed_experts=16,
         intermediate_size=64, rel_extent=64, sliding_window_size=64, log_scaling_n_floor=32)
cfg.pop("vision_config", None)
args = inkling2.ModelArgs.from_dict(cfg)
mx.random.seed(0)
model = inkling2.Model(args)
model.update(nn.utils.tree_map(lambda p: (mx.random.normal(p.shape) * 0.05).astype(mx.bfloat16)
                               if p.dtype == mx.bfloat16 or p.dtype == mx.float32 else p,
                               model.parameters()))
nn.quantize(model, group_size=16, bits=4, mode="nvfp4",
            class_predicate=lambda p, m: ".experts." in p and hasattr(m, "to_quantized"))
mx.eval(model.parameters())

def run(B, ctx):
    cache = make_prompt_cache(model)
    prompt = mx.random.randint(0, 500, (B, ctx))
    mx.eval(model(prompt, cache=cache))
    y = mx.random.randint(0, 500, (B, 1))
    for _ in range(3):
        mx.eval(model(y, cache=cache))
    out = model(y, cache=cache)
    mx.export_to_dot("/tmp/fuse_opcount.dot", out)
    mx.eval(out)
    lines = [l for l in open("/tmp/fuse_opcount.dot") if "label" in l and "->" not in l]
    labels = [re.search(r'label\s*=\s*"([^"]*)"', l).group(1) for l in lines if "rectangle" in l]
    labels = (len(lines), labels)
    builds = []
    for _ in range(9):
        t0 = time.perf_counter(); o = model(y, cache=cache); builds.append(time.perf_counter() - t0); mx.eval(o)
    return labels, sorted(builds)[4] * 1e3

print(f"mlx_lm from {mlx_lm.__file__}")
for B in (1, 4):
    labels, ms = run(B, 40)
    total, labels = labels
    print(f"B={B}: nodes={total} prims={len(labels)}  build={ms:.1f} ms")
    if B == 1 and "-v" in sys.argv:
        for k, v in collections.Counter(re.sub(r"\d+$", "", l) for l in labels).most_common(40):
            print(f"   {v:6d} {k}")

if "--dump" in sys.argv:
    # Greedy decode on the random model; save the logits for a base/fuse diff.
    import numpy as np
    path = sys.argv[sys.argv.index("--dump") + 1]
    res = {}
    for B in (1, 3):
        cache = make_prompt_cache(model)
        mx.random.seed(1)
        prompt = mx.random.randint(0, 500, (B, 30))
        logits = model(prompt, cache=cache)[:, -1]
        steps = [logits]
        for _ in range(24):
            y = mx.argmax(logits, axis=-1)[:, None]
            logits = model(y, cache=cache)[:, -1]
            steps.append(logits)
        res[f"B{B}"] = np.array(mx.stack(steps, 1))
    np.savez(path, **res)
    print("saved", path)
