#!/bin/bash
# Evaluate thermal checkpoints of new runs on B as soon as A marks them DONE. Waits for the earlier B chains first.
export CVR_ROOT=/root/cvresearch CVR_DATA=/root/cvresearch/data OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /root/cvresearch/ViPT
PY=/root/miniconda3/envs/sdstrack/bin/python
CK=output/checkpoints/train/vipt
LOG=/root/cvresearch/B_daemon3.log
while ! grep -q "^END" /root/cvresearch/B_daemon.log 2>/dev/null; do sleep 60; done
echo "START $(date)" > $LOG
job() { name=$1; shift   # then pairs src:yaml
  while [ ! -f /mnt/cvr_eval/ckpt_$name/DONE ]; do sleep 60; done
  for p in "$@"; do src=${p%%:*}; y=${p##*:}
    mkdir -p $CK/$y RGBT_workspace/results/LasHeR/$y; cp /mnt/cvr_eval/ckpt_$name/$src $CK/$y/ViPTrack_ep0120.pth.tar; cp -n experiments/vipt/batch1_modality.yaml experiments/vipt/$y.yaml
    while pgrep -f "[t]est_rgbt_mgpus" > /dev/null; do sleep 30; done
    echo "[$(date +%T)] START $y" >> $LOG
    $PY RGBT_workspace/test_rgbt_mgpus.py --script_name vipt --dataset_name LasHeR --yaml_name $y --epoch 120 --mode parallel --threads 4 --seq_home /root/cvresearch/data/lasher/testingset > /root/cvresearch/B_eval_$y.log 2>&1
    echo "[$(date +%T)] END $y files=$(ls RGBT_workspace/results/LasHeR/$y | wc -l)" >> $LOG
  done; }
job batch9_indep_thermal_norgb ViPTrack_ep0120.pth.tar:B_indep_thermal_norgb
echo "END $(date)" >> $LOG
