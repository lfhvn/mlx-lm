#!/bin/bash
# Bench base then a snapshot of one commit of this branch, back to back,
# each under the GPU lock. The snapshot lets the worktree change meanwhile.
# usage: bench/bench_pair.sh <label> [rev] [bench args...]
set -e
cd "$(dirname "$0")/.."
label=$1; rev=${2:-HEAD}; shift 2 || shift $#
snap=/private/tmp/claude-501/-Users-lhm/f1d7e6fe-edea-489b-a52e-ed9bed562c7d/scratchpad/snap_$label
rm -rf "$snap"; mkdir -p "$snap"; git archive "$rev" mlx_lm | tar -x -C "$snap"
common=/Users/lhm/inkling-orchestra/engine/common
python3 $common/gpulock.py ../mlx-lm-base/.venv/bin/python -I $common/bench_decode.py "$@" > bench/runs/${label}_base.txt 2>&1
python3 $common/gpulock.py .venv/bin/python -I -c "import sys,runpy; sys.path.insert(0,'$snap'); sys.argv=['bench']+sys.argv[1:]; runpy.run_path('$common/bench_decode.py', run_name='__main__')" "$@" > bench/runs/${label}_fuse.txt 2>&1
paste bench/runs/${label}_base.txt bench/runs/${label}_fuse.txt
