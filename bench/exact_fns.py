"""Each compiled helper vs the same code with compile off, on large inputs."""
import mlx.core as mx
from mlx_lm.models.inkling2 import layers as L

mx.random.seed(0)
N = 1 << 14
bf = mx.bfloat16
def r(*s, dt=mx.float32, sc=3.0): return (mx.random.normal(s) * sc).astype(dt)
cases = {
    "_l2_clip": lambda: (L._l2_clip, (r(64, N // 64, 128, dt=bf, sc=4), r(N // 64, 128, sc=1), 22.6)),
    "_l2_clip_tau": lambda: (L._l2_clip_tau, (r(64, N // 64, 128, dt=bf, sc=4), r(N // 64, 128), 22.6, 1 + mx.abs(r(64, 1, 1, sc=.1)))),
    "_softcapped_swiglu": lambda: (L._softcapped_swiglu, (r(64, N, dt=bf), r(64, N, dt=bf, sc=10), 4.0, 25.0)),
    "_output_gate": lambda: (L._output_gate, (r(64, N, dt=bf), r(64, N, dt=bf))),
    "_conv_residual": lambda: (L._conv_residual, (r(64, N, dt=bf), r(64, N, dt=bf), 5 ** .5)),
    "_message_mix": lambda: (L._message_mix, (r(64, N), r(64, N), r(64, N), r(64, N))),
    "_f32_add": lambda: (L._f32_add, (r(64, N, dt=bf), r(64, N))),
    "_weighted_sum": lambda: (L._weighted_sum, (r(64, 8, N, dt=bf), mx.abs(r(64, 8)))),
    "_dynamic_taps": lambda: (L._dynamic_taps, (r(64, 8, N // 8, 4, dt=bf), r(8, N // 8, 4), r(64, 8, 1, 4, dt=bf))),
}
for name, mk in cases.items():
    for trial in range(4):
        f, args = mk()
        got = f(*args)
        mx.disable_compile(); ref = f(*args); mx.enable_compile()
        d = mx.abs(got.astype(mx.float32) - ref.astype(mx.float32))
        n = int((d > 0).sum())
        if n or trial == 3:
            print(f"{name}: {n} of {got.size} differ, max {float(d.max()):.3g}")
            break
