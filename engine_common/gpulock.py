"""Run a command while holding the shared GPU lock, so only one engine
experiment loads a model / benchmarks at a time.

    python3 gpulock.py <command> [args...]

Waits for the lock, then waits until system memory is at least 35% free
(two experiment servers are running on this machine), then runs the command.
"""
import fcntl, subprocess, sys, time, re

LOCK = "/Users/lhm/inkling-orchestra/engine/.gpu.lock"

def free_pct():
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    m = re.search(r"free percentage: (\d+)%", out)
    return int(m.group(1)) if m else 100

with open(LOCK, "w") as f:
    t = time.time()
    fcntl.flock(f, fcntl.LOCK_EX)
    waited = time.time() - t
    while free_pct() < 35:
        time.sleep(5)
    print(f"[gpulock] acquired after {waited:.0f}s, memory free {free_pct()}%", file=sys.stderr, flush=True)
    sys.exit(subprocess.call(sys.argv[1:]))
