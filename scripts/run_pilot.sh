#!/usr/bin/env bash
# Pilot: four federated variants, identical seed, so the comparison is paired.
#
# Purpose is EFFECT SIZE, not paper numbers. 20 rounds is ~2 epochs of the
# reproduction's data budget, and it compresses the 120-round warm-up schedule,
# so `ours` vs `modality` is not a fair test here. What it can answer is whether
# modality-aware aggregation separates from plain FedAvg at all, and by how much
# -- which is what the experiment matrix decision needs.
set -u

PY="D:/Conda/envs/sdstrack/python.exe"
OUT="D:/cvresearch/ViPT/output/fed/pilot20"
ROUNDS=20
STEPS=15
SEED=42

cd "D:/cvresearch/ViPT" || exit 1
mkdir -p "$OUT"

echo "pilot start $(date '+%F %T')  rounds=$ROUNDS local_steps=$STEPS seed=$SEED"
echo ""

for V in local fedavg modality ours; do
  echo "================ $V  $(date '+%T') ================"
  t0=$(date +%s)
  "$PY" lib/federated/server.py \
      --variant "$V" --rounds "$ROUNDS" --local-steps "$STEPS" --seed "$SEED" \
      --out-dir "$OUT/$V" --save-every 10 --export-at-end \
      > "$OUT/${V}.log" 2>&1
  rc=$?
  t1=$(date +%s)
  echo "$V finished rc=$rc in $(( (t1-t0)/60 )) min"
  if [ $rc -ne 0 ]; then
    echo "!! $V FAILED -- last 20 lines:"
    tail -20 "$OUT/${V}.log"
  fi
  echo ""
done

echo "pilot done $(date '+%F %T')"
