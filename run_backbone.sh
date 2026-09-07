#!/usr/bin/env bash
# run_backbone.sh <enc> <main_gpu> <aux_gpu>        e.g.  ./run_backbone.sh jina5s 0 1
#
# The whole Pilot M chain (notes/backbones.md) for one backbone on its own pair of cards,
# resumable by markers under $DVLSR_DATA/markers/bb_<enc>/. Nothing here waits on another
# backbone: run one instance per backbone on disjoint card pairs.
#
# Needs: `source env.sh` values (see env.example.sh), the §0 artifacts listed in
# notes/backbones.md "Running on another server", and the HF models cached or reachable.
# Tunables: BB_PROTO_SHARDS (default 4, 2 per card), BB_PROTO_BS (64), BB_MAX_STEPS (unset =
# the full 3,124 steps used for e5).
set -uo pipefail
ENC=$1; G=$2; AUX=$3
REPO=$(cd "$(dirname "$0")" && pwd); cd "$REPO"
for f in "$REPO/env.sh" "${DVLSR_ENV:-}"; do [ -n "$f" ] && [ -f "$f" ] && source "$f"; done
: "${DVLSR_DATA:?set DVLSR_DATA (see env.example.sh)}"
M=$DVLSR_DATA/markers/bb_$ENC; mkdir -p "$M" "$DVLSR_DATA/markers"
# results/logs directories exactly as dvlsr/paths.py resolves them (absolute or relative)
RES=$(python -c "from dvlsr import paths; print(paths.RESULTS)")
L=$(python -c "from dvlsr import paths; print(paths.LOGS)"); mkdir -p "$L"
step () {  # step <marker> <cmd...>
  local mk=$1; shift
  [ -f "$M/$mk" ] && { echo "[$(date +%H:%M)] skip $mk"; return 0; }
  echo "[$(date +%H:%M)] start $mk"
  if "$@" > "$L/bb_${ENC}_$mk.log" 2>&1; then touch "$M/$mk"; echo "[$(date +%H:%M)] done  $mk"
  else echo "[$(date +%H:%M)] FAIL  $mk (see $L/bb_${ENC}_$mk.log)"; return 1; fi
}
export CUDA_VISIBLE_DEVICES=$G
step A1_banks python scripts/10_probes_banks.py --encoder $ENC || exit 1
step A2_aux   python scripts/10_probes_banks.py --encoder $ENC --aux || exit 1
step A3_r1    python scripts/12_r1.py --encoder $ENC || exit 1
export CUDA_VISIBLE_DEVICES=$G,$AUX
step B_protos python scripts/11_prototypes.py --encoder $ENC --n-shards ${BB_PROTO_SHARDS:-4} \
  --per-gpu 2 --bs ${BB_PROTO_BS:-64} || exit 1
export CUDA_VISIBLE_DEVICES=$G
step C1_whiten python scripts/13_whitening.py --encoder $ENC || exit 1
step C2_probe  python scripts/95_layer_probe.py --encoder $ENC || exit 1
LAYER=$(python -c "import json;print(json.load(open('$RES/95_layer_probe_$ENC.json'))['layer'])")
TAU=$(python -c "import json;print(json.load(open('$RES/95_layer_probe_$ENC.json'))['tau'])")
echo "[$(date +%H:%M)] $ENC: layer $LAYER tau $TAU"
# ---- training on the main card; domain vocabularies and the dense reference on the aux card ----
( export CUDA_VISIBLE_DEVICES=$G
  step D_train python scripts/42_train.py --name V1oracle_$ENC --encoder $ENC --layer $LAYER --tau $TAU \
    --oracle --split none --rep R2 --r1-prefix none --lam-d 3e-4 --lam-q 9e-4 --lr-head 2e-4 \
    --epochs 2 --batch-queries 128 --warmup 500 --enc-chunk 128 --rep-chunk 256 --seed 1 \
    --min-free-gb 0 ${BB_MAX_STEPS:+--max-steps $BB_MAX_STEPS} ) &
TRAIN_PID=$!
( export CUDA_VISIBLE_DEVICES=$AUX
  for C in scifact nfcorpus trec-covid; do
    (  # backbone drivers share the per-corpus vocabulary and reference files
      flock 8
      step E1_vocab_${C}       python scripts/91_domain_vocab.py --name $C --encoder $ENC || exit 1
      step E1_vocab_${C}_tfidf python scripts/91_domain_vocab.py --name $C --encoder $ENC --select tfidf || exit 1
      step E1_vocab_${C}_dfcap python scripts/91_domain_vocab.py --name $C --encoder $ENC --select dfcap || exit 1
      step E2_dense_${C}       python scripts/94_domain_baselines.py --name $C --encoder $ENC --skip-splade || exit 1
    ) 8> "$DVLSR_DATA/markers/domain_$C.lock" || exit 1
  done ) &
AUX_PID=$!
wait $TRAIN_PID || { echo "[$(date +%H:%M)] training failed"; exit 1; }
wait $AUX_PID   || { echo "[$(date +%H:%M)] domain vocabulary stage failed"; exit 1; }
# ---- evaluations: trec-covid (the long pole) on the main card, the small corpora on aux ----
evals () {  # evals <corpus>
  local C=$1 other
  other=$([ "$C" = scifact ] && echo nfcorpus || echo scifact)
  step E3_${C}_df    python scripts/92_domain_eval.py --name $C --name-model V1oracle_$ENC || return 1
  step E3_${C}_tfidf python scripts/92_domain_eval.py --name $C --name-model V1oracle_$ENC --select tfidf || return 1
  step E3_${C}_dfcap python scripts/92_domain_eval.py --name $C --name-model V1oracle_$ENC --select dfcap || return 1
  step E3_${C}_norm  python scripts/92_domain_eval.py --name $C --name-model V1oracle_$ENC --norm || return 1
  step E3_${C}_rand  python scripts/92_domain_eval.py --name $C --name-model V1oracle_$ENC --control rand || return 1
  step E3_${C}_ctl   python scripts/92_domain_eval.py --name $C --name-model V1oracle_$ENC --control $other || return 1
}
( export CUDA_VISIBLE_DEVICES=$G;   evals trec-covid ) & P1=$!
( export CUDA_VISIBLE_DEVICES=$AUX; evals scifact && evals nfcorpus ) & P2=$!
wait $P1 || { echo "[$(date +%H:%M)] trec-covid evaluations failed"; exit 1; }
wait $P2 || { echo "[$(date +%H:%M)] small-corpus evaluations failed"; exit 1; }
python scripts/93_domain_report.py > "$L/bb_${ENC}_report.log" 2>&1 || echo "[$(date +%H:%M)] report failed (see $L/bb_${ENC}_report.log)"
echo "[$(date +%H:%M)] BACKBONE_DONE $ENC"
