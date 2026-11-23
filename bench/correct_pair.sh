#!/bin/bash
# Real-model logits/greedy dump for base and a snapshot of one commit, then diff.
# usage: bench/correct_pair.sh <label> [rev]
set -e
cd "$(dirname "$0")/.."
label=$1; rev=${2:-HEAD}
snap=/private/tmp/claude-501/-Users-lhm/f1d7e6fe-edea-489b-a52e-ed9bed562c7d/scratchpad/snap_$label
rm -rf "$snap"; mkdir -p "$snap"; git archive "$rev" mlx_lm | tar -x -C "$snap"
out=/private/tmp/claude-501/-Users-lhm/f1d7e6fe-edea-489b-a52e-ed9bed562c7d/scratchpad
common=/Users/lhm/inkling-orchestra/engine/common
here=$PWD
[ -f $out/correct_base.npz ] || python3 $common/gpulock.py ../mlx-lm-base/.venv/bin/python -I $here/bench/correctness.py $out/correct_base.npz
python3 $common/gpulock.py .venv/bin/python -I -c "import sys,runpy; sys.path.insert(0,'$snap'); sys.argv=['c','$out/correct_$label.npz']; runpy.run_path('$here/bench/correctness.py', run_name='__main__')"
.venv/bin/python -I bench/correctness.py --cmp $out/correct_base.npz $out/correct_$label.npz | tee bench/runs/correct_$label.txt
