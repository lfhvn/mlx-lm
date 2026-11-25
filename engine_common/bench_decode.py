"""Decode benchmark for one mlx-lm checkout. Run it with that checkout's
venv python, under gpulock.py, back to back with the base for comparison:

    python3 ../common/gpulock.py ../mlx-lm-base/.venv/bin/python -I ../common/bench_decode.py
    python3 ../common/gpulock.py .venv/bin/python -I ../common/bench_decode.py

Prints: ops per B=1 decode step, Python graph-build ms, and ms/step + tok/s
at several batch sizes, with a short and a ~4k-token context. Absolute
numbers drift with the other jobs on the GPU; compare runs made back to back.
"""
import argparse, time
import mlx.core as mx
import mlx_lm
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.models.cache import make_prompt_cache

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="pipenetwork/Inkling-2-Mini-MLX-bf16")
ap.add_argument("--batches", default="1,4,8")
ap.add_argument("--steps", type=int, default=40)
args = ap.parse_args()
print(f"mlx_lm from {mlx_lm.__file__}")
model, tok = load(args.model)
short = tok.encode("def fib(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n")
filler = " ".join(f"step {i} checked the edge case and found it fine." for i in range(1500))
long = tok.encode(filler)[:4000] + short

cache = make_prompt_cache(model)
mx.eval(model(mx.array([short]), cache=cache))
y = mx.array([[short[-1]]])
out = model(y, cache=cache)
mx.export_to_dot("/tmp/bench_step.dot", out); mx.eval(out)
ops = sum(1 for l in open("/tmp/bench_step.dot") if "label" in l and "->" not in l)
builds = []
for _ in range(10):
    t = time.perf_counter(); o = model(y, cache=cache); builds.append(time.perf_counter() - t); mx.eval(o)
print(f"ops per decode step (B=1): {ops}")
print(f"graph build per step (B=1): {sorted(builds)[5]*1e3:.1f} ms")

def step_ms(prompt, B, steps):
    gen = BatchGenerator(model, max_tokens=steps + 20, stop_tokens=None)
    gen.insert([prompt] * B, [steps + 20] * B)
    n = 0
    while gen.next_generated():
        n += 1
        if n == 10:
            mx.synchronize(); t0 = time.perf_counter()
        if n == 10 + steps:
            mx.synchronize(); break
    gen.close()
    return (time.perf_counter() - t0) / steps * 1e3

for name, prompt in [("short", short), ("4k", long)]:
    for B in [int(b) for b in args.batches.split(",")]:
        ms = step_ms(prompt, B, args.steps)
        print(f"ctx={name:5s} B={B:2d}: {ms:6.1f} ms/step  {B*1000/ms:6.1f} tok/s total")
