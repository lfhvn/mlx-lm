#!/bin/zsh
# Real-model checks of engine-combined against engine-base, in one GPU-lock hold:
# correctness (compare_engines.py), memory (repro_mem.py), speed (bench_decode.py).
C=/Users/lhm/inkling-orchestra/engine/common
BASE=/Users/lhm/inkling-orchestra/engine/mlx-lm-base/.venv/bin/python
NEW=/Users/lhm/inkling-orchestra/engine/combined/.venv/bin/python
OUT=/Users/lhm/inkling-orchestra/engine/combined/verify
mkdir -p $OUT
f() { grep -v -E "Warning|Fetching|warn" ; }
echo "== correctness"; $BASE -I $C/compare_engines.py --out $OUT/cmp_base.json 2>&1 | f
$NEW -I $C/compare_engines.py --out $OUT/cmp_new.json 2>&1 | f
python3 $C/compare_engines.py --compare $OUT/cmp_base.json $OUT/cmp_new.json | tee $OUT/compare.txt
echo "== memory"; $NEW -I $C/repro_mem.py --batch 4 --steps 1500 --every 250 2>&1 | f | tee $OUT/mem_new.txt
echo "== speed"; $BASE -I $C/bench_decode.py --batches 1,4,8,16 2>&1 | f | tee $OUT/bench_base.txt
$NEW -I $C/bench_decode.py --batches 1,4,8,16 2>&1 | f | tee $OUT/bench_new.txt
