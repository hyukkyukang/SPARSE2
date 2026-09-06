#!/usr/bin/env bash
# Follow-up experiments after the pilot study, on dslab-gpu10 (5x TITAN RTX 24 GB).
#
# The study's artifacts lived on the original A100 box, so the §0 setup is rebuilt
# here first (see notes/deviations.md D12: no full-collection dense encode, C1 built
# without the dense candidates, fp16 autocast). Every stage is resumable; select
# stages with STAGES="refs art pilotC ...".
set -uo pipefail
D=${DVLSR_DATA:-/mnt/sdc/hkkang/dvlsr}
source $D/env.sh
cd "$(dirname "$0")"
LOG=$D/logs/followups.log
ENC="--n-shards 10 --per-gpu 2"                      # artifact-stage encode sharding
NG=$(python -c "import os;print(len(os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')))")
PD="--enc-shards $((NG*2)) --enc-per-gpu 2 --enc-chunk 512 --train-per-gpu ${TRAIN_PER_GPU:-2}"
BQ=${BQ:-128}                                         # V1 batch (queries); probed below

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $LOG; }
run() {  # run <marker> <cmd...>: skip when the marker exists, write it on success.
         # A per-step lock lets two driver instances (e.g. STAGES=refs and STAGES=art)
         # share the same markers without duplicating work.
  local m=$D/markers/$1; shift
  mkdir -p $D/markers
  (
    flock -x 9
    if [ -f "$m" ]; then say "skip  $(basename $m)"; exit 0; fi
    say "start $(basename $m): $*"
    if "$@" >> $D/logs/$(basename $m).log 2>&1; then touch "$m"; say "done  $(basename $m)"; exit 0
    else say "FAIL  $(basename $m) (see $D/logs/$(basename $m).log)"; exit 1; fi
  ) 9>"$m.lock"
}
need() {  # need <marker...>: block until prerequisite steps (possibly of another instance) are done
  for m in "$@"; do
    [ -f $D/markers/$m ] || say "waiting for $m"
    until [ -f $D/markers/$m ]; do sleep 60; done
  done
}

stage_refs() {   # reference systems, C1, negatives
  run 03_refs        python scripts/03_reference_runs.py --what all || return 1
  run 05_c1          python scripts/05_build_c1.py || return 1
  run 06_bm25_c1     python scripts/06_references_c1.py --what bm25 || return 1
  # one worker per card: these overlap with the artifact stage's prototype workers
  run 06_splade_c1   python scripts/06_references_c1.py --what splade --n-shards 10 --per-gpu 1 || return 1
  run 02_dense_c1    python scripts/02_encode_corpus.py --encoder e5 --subset c1 --n-shards 10 --per-gpu 1 --bs 256 || return 1
  run 06_dense_c1    python scripts/06_references_c1.py --what dense --encoder e5 || return 1
  run 40_negs        python scripts/40_mine_negatives.py --threads 48 || return 1
}
stage_art() {    # probes, banks, prototypes, R1, whitening (e5 only)
  run 10_plan        python scripts/10_probes_banks.py --encoder plan || return 1
  run 10_e5          python scripts/10_probes_banks.py --encoder e5 || return 1
  run 10_e5_aux      python scripts/10_probes_banks.py --encoder e5 --aux || return 1
  run 16_splade_exp  python scripts/16_splade_expansions.py || return 1
  run 11_proto_e5    python scripts/11_prototypes.py --encoder e5 $ENC --bs 128 || return 1
  run 12_r1_e5       python scripts/12_r1.py --encoder e5 || return 1
  run 13_whiten_e5   python scripts/13_whitening.py --encoder e5 || return 1
}
stage_pilotC() { # training-free retrieval at layer 12: R2 (full grid) and R1 (the bare-string check)
  need 06_dense_c1 10_e5_aux 16_splade_exp 13_whiten_e5
  run 30_c1_L12_R2   python scripts/30_encode_c1.py --encoder e5 --layer 12 --rep R2 --transform whitened --r1-prefix none $ENC --bs 128 || return 1
  run 31_L12_R2      python scripts/31_pilotC.py --encoder e5 --layer 12 --rep R2 --r1-prefix none || return 1
  run 32_decision    python scripts/32_pilotC_decision.py --file 31_pilotC_e5_L12_R2.json || return 1
  run 30_c1_L12_R1   python scripts/30_encode_c1.py --encoder e5 --layer 12 --rep R1 --transform whitened --r1-prefix none $ENC --bs 128 || return 1
  run 31_L12_R1      python scripts/31_pilotC.py --encoder e5 --layer 12 --rep R1 --r1-prefix none --fast || return 1
}
probe_batch() {  # find the largest V1 batch that fits a 24 GB card
  for b in $BQ 96 64; do
    say "probe V1 batch $b"
    if CUDA_VISIBLE_DEVICES=0 python scripts/42_train.py --name probe --layer 12 \
         --tau $(python -c "import json;print(json.load(open('${DVLSR_RESULTS_DIR:-results}/32_pilotC_decision.json'))['tau'])") \
         --split cluster --batch-queries $b --enc-chunk 512 --max-steps 4 --warmup 2 \
         > $D/logs/probe_batch_$b.log 2>&1; then BQ=$b; say "V1 batch = $b"; rm -rf $D/ckpt/probe; return 0; fi
  done
  say "no batch size fits"; return 1
}
pd() {  # pd <group-marker> <names>: train, encode, evaluate a group of Pilot D runs
  local g=$1 names=$2
  run pd_train_$g  python scripts/45_run_pilotD.py --stage train  --only $names $PD --batch-queries $BQ || return 1
  run pd_enc_$g    python scripts/45_run_pilotD.py --stage encode --only $names $PD || return 1
  run pd_eval_$g   python scripts/45_run_pilotD.py --stage eval   --only $names $PD || return 1
  python scripts/46_pilotD_report.py > /dev/null 2>&1 || true
}
stage_pilotD_cluster() {   # the semantic-dropout question, on the OOD split
  need 40_negs
  run 41_splits python scripts/41_vocab_splits.py --encoder e5 --layer 12 --rep R2 || return 1
  probe_batch || return 1
  pd A "V1oracle_cluster,V1_cluster,V1vd_cluster,V1cvd_cluster,V1meta_cluster,V1lin_cluster"
}
stage_pilotE() {           # insertion-time calibration on the cluster split
  # one worker per card: four C1 encodes per variant, and an OOM here wastes the whole variant
  run 51_pilotE_V1_cluster python scripts/51_run_pilotE.py --name V1_cluster --enc-shards 8 --enc-per-gpu 1
}
stage_pilotD_R1() {        # bare-string entries, trained (they beat prototypes untrained at layer 12)
  pd R "V1oracle_R1,V1_R1,V1oracle_cluster_R1,V1_cluster_R1,V1cvd_cluster_R1"
}
stage_pilotD_Wh() {        # entry whitening from the token bank instead of the seen entries
  pd W "V1oracle_cluster_Wh,V1_cluster_Wh"
}
stage_pilotF() {           # per-entry background normalisation inside the score
  pd F1 "V1oracle_norm_cluster,V1norm_cluster"
  pd F2 "V1oracle_normz_cluster,V1normz_cluster"
  pd F3 "V1oracle_normWh_cluster,V1normWh_cluster"
  pd F4 "V1oracle_norm,V1norm"
}
stage_pilotG() {           # what the Pilot F result makes worth testing
  pd G1 "V1oracle_norm_cluster_R1,V1norm_cluster_R1"     # normalisation rescues bare strings?
  pd G2 "V1oracle_norm_rare,V1norm_rare"                 # does it damage the motivating case?
  pd G3 "V1oracle_param,V1param"                         # cost of a text-defined vocabulary
}
stage_pilotD_random() {    # baselines, seeds, capacity, distillation ceiling, entry transform
  pd B "V1oracle,V1,V1vd,V1cvd,V1meta,V1lin,V1dist_oracle,V1dist,Crand,V1_s2,V1vd_s2,V1oracle_Wh,V1_Wh"
}
stage_pilotD_rare() {
  pd C "V1oracle_rare,V1_rare,V1vd_rare"
}
stage_phrase() {           # entity / phrase insertion
  run 80_phrase_vocab python scripts/80_phrase_vocab.py || return 1
  run 81_phrase_proto python scripts/81_phrase_prototypes.py --encoder e5 $ENC || return 1
  run 82_phrase_ckpt  python scripts/82_phrase_ckpt.py || return 1
  pd P "V1oracle_phrase,V1_phrase"
}
stage_pilotH() {           # seed repeats for the arms the decision now rests on
  pd H "V1_cluster_s2,V1norm_cluster_s2"
}
stage_pilotI() {           # a shared learned map on the entry side
  pd I1 "V1oracle_eh_cluster,V1eh_cluster"
  pd I2 "V1oracle_eh,V1eh"
  pd I3 "V1oracle_ehnorm_cluster,V1ehnorm_cluster"
}
stage_pilotJ() {           # k=100 prototypes: is the residual gap prototype noise?
  pd J "V1oracle_k100_cluster,V1k100_cluster"
}
stage_pilotD_enc() {       # encoder-training arms (slowest; last)
  # V2 (LoRA without vocabulary dropout) is the collapse case; it belongs here so the
  # dropout effect can be stated from this rebuild rather than from the A100 logs
  pd F "V3oracle,V3,V3vd,V2vd,V2,V3_cluster"
}

STAGES=${STAGES:-"refs art pilotC pilotD_cluster pilotE pilotD_R1 pilotD_Wh pilotF pilotG pilotH pilotI pilotJ pilotD_random pilotD_rare phrase pilotD_enc"}
say "===== run_followups: stages [$STAGES] ====="
for s in $STAGES; do
  say "----- stage $s -----"
  stage_$s || { say "stage $s failed; stopping"; exit 1; }
done
say "ALL_STAGES_DONE"
