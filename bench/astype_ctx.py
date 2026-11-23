"""Group the AsType nodes of /tmp/fuse_opcount.dot by producer and consumer ops."""
import re, collections, sys
lab, prod, cons = {}, {}, collections.defaultdict(list)
ins = collections.defaultdict(list)
for l in open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/fuse_opcount.dot"):
    m = re.search(r'\{ (\d+) \[label ="([^"]*)"', l)
    if m: lab[m.group(1)] = m.group(2); continue
    m = re.search(r'"?(\w+)"? -> "?(\w+)"?', l)
    if not m: continue
    a, b = m.groups()
    if a.isdigit(): prod[b] = a          # prim -> array
    else: cons[a].append(b); ins[b].append(a)  # array -> prim
c = collections.Counter()
want = sys.argv[2] if len(sys.argv) > 2 else "AsType"
for n, v in lab.items():
    if v != want: continue
    src = [lab.get(prod.get(a), "input") for a in ins[n]]
    outs = [a for a, p in prod.items() if p == n]
    dst = sorted({lab[q][:40] for a in outs for q in cons[a]})
    c[(tuple(src), tuple(dst))] += 1
for k, v in c.most_common(40): print(v, k)
