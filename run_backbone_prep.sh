#!/usr/bin/env bash
# run_backbone_prep.sh <enc> <gpu_a> <gpu_b>     e.g.  ./run_backbone_prep.sh arctic 0 2
#
# The §0 artifact stages of run_backbone.sh for one backbone -- banks, aux, R1, prototypes,
# whitening and the layer probe -- without the pilot's training and domain evaluations.
# Resumable by the same markers under $DVLSR_DATA/markers/bb_<enc>/.
set -uo pipefail
ENC=$1; G=$2; AUX=$3
REPO=$(cd "$(dirname "$0")" && pwd); cd "$REPO"
for f in "$REPO/env.sh" "${DVLSR_ENV:-}"; do [ -n "$f" ] && [ -f "$f" ] && source "$f"; done
: "${DVLSR_DATA:?set DVLSR_DATA (see env.example.sh)}"
M=$DVLSR_DATA/markers/bb_$ENC; mkdir -p "$M"
RES=$(python -c "from dvlsr import paths; print(paths.RESULTS)")
L=$(python -c "from dvlsr import paths; print(paths.LOGS)"); mkdir -p "$L"
step () { local mk=$1; shift
  [ -f "$M/$mk" ] && { echo "[$(date +%H:%M)] skip $mk"; return 0; }
  echo "[$(date +%H:%M)] start $mk"
  if "$@" > "$L/bb_${ENC}_$mk.log" 2>&1; then touch "$M/$mk"; echo "[$(date +%H:%M)] done  $mk"
  else echo "[$(date +%H:%M)] FAIL  $mk (see $L/bb_${ENC}_$mk.log)"; return 1; fi; }
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
python -c "import json;d=json.load(open('$RES/95_layer_probe_$ENC.json'));print('[prep] $ENC: layer',d['layer'],'tau',d['tau'])"
echo "[$(date +%H:%M)] PREP_DONE $ENC"
