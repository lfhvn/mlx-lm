"""Verify-forward cost with the prefill kernel (DECODE_ROWS=0) vs per-query
decode attention (DECODE_ROWS=8), short and 4k context."""
import sys, time
sys.path.insert(0, "/Users/lhm/inkling-orchestra/engine/loop-spec")
import mlx.core as mx
from mlx_lm import load
import mlx_lm.models.inkling2.layers as layers
from loopspec.spec import make_cache, prefill, forward
from loopspec.prompts import chat_prompts, long_prompt

model, tok = load("/Users/lhm/models/TML-d24-varloop-mlx")
for name, ctx in [("short", chat_prompts(tok)[0]), ("4k", long_prompt(tok))]:
    for passes in (1, 3):
        c = make_cache(model, passes, True)
        prefill(model, passes, ctx[:-8], c)
        for rows in (0, 8):
            layers.DECODE_ROWS = rows
            for L in (1, 2, 4, 7):
                ts = []
                for _ in range(8):
                    mx.synchronize(); t = time.perf_counter()
                    out = forward(model, passes, mx.array(ctx[-8:-8 + L]), c)
                    mx.eval(out)
                    mx.synchronize(); ts.append(time.perf_counter() - t)
                    for cc in c:
                        cc[1].rollback(0); cc[0].trim(L)
                if L == 7:
                    if rows == 0:
                        ref = out
                    else:
                        d = mx.abs(out - ref).max().item()
                        am = (mx.argmax(out[0], -1) == mx.argmax(ref[0], -1)).all().item()
                print(f"ctx={name} passes={passes} rows={rows} L={L}: {sorted(ts)[3]*1e3:.1f} ms", flush=True)
        print(f"ctx={name} passes={passes} L=7 kernel vs rows: max logit diff {d:.3f}, same argmax {am}")
        layers.DECODE_ROWS = 0
        del c
        mx.clear_cache()
