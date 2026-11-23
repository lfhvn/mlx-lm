"""Acceptance predicted from teacher-forced agreement (agree.json)."""
import json, sys
rows = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "loopspec/out/agree.json"))
for p in ("1", "2", "5", "3"):
    m = sum((r["match"][p] for r in rows), [])
    print(f"passes {p}: agreement {sum(m)/len(m):.3f} over {len(m)} tokens")
for p in ("1", "2"):
    for k in range(2, 7):
        toks = rounds = acc = 0
        for r in rows:
            v = r["match"][p]; t = 0
            while t < len(v):
                a = 0
                while a < k and t + a < len(v) and v[t + a]:
                    a += 1
                rounds += 1; acc += a; t += a + 1; toks += a + 1
        print(f"passes {p} k={k}: mean accept {acc/rounds:.2f}, tok/round {toks/rounds:.2f}")
