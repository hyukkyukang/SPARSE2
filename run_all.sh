#!/usr/bin/env bash
# Full pilot study, in the order the protocol requires (A -> B -> C -> D -> E).
# Large artifacts go to $DVLSR_DATA (default /workspace/SPARSE/dvlsr).
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.
export OPENBLAS_NUM_THREADS=32 OMP_NUM_THREADS=32   # 256 cores > OpenBLAS's 128 limit
export JAVA_HOME=${JAVA_HOME:-/workspace/SPARSE/dvlsr/tools/jdk-21.0.5+11}
export PATH=$JAVA_HOME/bin:$PATH
export PYSERINI_CACHE=${PYSERINI_CACHE:-/workspace/SPARSE/dvlsr/pyserini_cache}

# ---------------------------------------------------------------- §0 setup
python3 -c "from dvlsr.data import build_collection_binary, build_splits; \
            build_collection_binary(); build_splits()"
python3 scripts/01_vocab.py                                   # freeze V (§0.3)
for E in e5 bge; do
  python3 scripts/02_encode_corpus.py --encoder $E --n-shards 28 --per-gpu 2
  python3 scripts/04_dense_retrieval.py --encoder $E          # sanity vs published
done
python3 scripts/03_reference_runs.py --what all               # BM25 + SPLADE++ (full)
python3 scripts/05_build_c1.py                                # C1 (§0.2)
python3 scripts/06_references_c1.py --what all --encoder e5   # references on C1

# ---------------------------------------------------------------- Pilot A
python3 scripts/10_probes_banks.py --encoder plan
python3 scripts/17_polysemy.py
python3 scripts/16_splade_expansions.py
for E in e5 bge colbert; do
  python3 scripts/10_probes_banks.py --encoder $E
  python3 scripts/10_probes_banks.py --encoder $E --aux
  python3 scripts/11_prototypes.py --encoder $E --n-shards 21 --per-gpu 3
  python3 scripts/12_r1.py --encoder $E
  python3 scripts/13_whitening.py --encoder $E
  python3 scripts/14_pilotA.py --encoder $E --mode prefix --reps R1 --no-overlap --no-sense
done
python3 scripts/14_pilotA.py --encoder e5      --mode report --r1-prefix none
python3 scripts/14_pilotA.py --encoder bge     --mode report --r1-prefix query
python3 scripts/14_pilotA.py --encoder colbert --mode report --r1-prefix none
python3 scripts/15_pilotA_report.py

# ---------------------------------------------------------------- Pilot B
python3 scripts/18_per_occurrence.py --encoder e5 --layers 9,10 --n-shards 12 --per-gpu 2
for L in 9 10; do
  python3 scripts/20_pilotB.py --encoder e5 --layer $L --store-layers 9,10 \
          --r1-prefix none --r3
done
python3 scripts/21_pilotB_report.py --files 20_pilotB_e5_L9.json 20_pilotB_e5_L10.json

# ---------------------------------------------------------------- Pilot C
run_c () {  # encoder layer rep transform [--fast]
  python3 scripts/30_encode_c1.py --encoder $1 --layer $2 --rep $3 --transform $4 \
          --r1-prefix none --n-shards 24 --per-gpu 3
  python3 scripts/31_pilotC.py --encoder $1 --layer $2 --rep $3 --transform $4 \
          --r1-prefix none ${5:-}
}
run_c e5  9 R2 whitened
run_c e5  9 R1 whitened --fast
run_c e5  9 R3 whitened --fast
run_c e5 12 R2 whitened --fast
run_c e5  9 R2 centered --fast
run_c bge 12 R2 whitened --fast
python3 scripts/33_qside_ablation.py
python3 scripts/32_pilotC_decision.py --file 31_pilotC_e5_L9_R2.json

# ---------------------------------------------------------------- Pilot D
python3 scripts/40_mine_negatives.py
python3 scripts/41_vocab_splits.py --encoder e5 --layer 9 --rep R2
python3 scripts/45_run_pilotD.py --stage lambda
python3 scripts/45_run_pilotD.py --stage train
python3 scripts/45_run_pilotD.py --stage encode
python3 scripts/45_run_pilotD.py --stage eval
python3 scripts/46_pilotD_report.py

# ---------------------------------------------------------------- Pilot E (conditional)
if python3 -c "import json,sys; \
   sys.exit(0 if json.load(open('results/46_pilotD_decision.json'))['decision']['pilotE_triggered'] else 1)"; then
  python3 scripts/51_run_pilotE.py --name V1
fi

# ---------------------------------------------------------------- confirmation
python3 scripts/60_full_corpus.py --name V1 --stage all
python3 scripts/70_synthesis.py
