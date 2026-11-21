#!/bin/bash
# Convergence curves (thermal): r20/r60/r100 of the batch-1 FedAvg and bucket runs. Runs only after B_daemon4 (zero-aux evals) has finished.
export CVR_ROOT=/root/cvresearch CVR_DATA=/root/cvresearch/data OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /root/cvresearch/ViPT
PY=/root/miniconda3/envs/sdstrack/bin/python
CK=output/checkpoints/train/vipt
LOG=/root/cvresearch/B_daemon5.log
while ! grep -q "^END" /root/cvresearch/B_daemon4.log 2>/dev/null; do sleep 60; done
echo "START $(date)" > $LOG
run() { y=$1
  while pgrep -f "[t]est_rgbt_mgpus" > /dev/null; do sleep 30; done
  [ -f $CK/$y/ViPTrack_ep0120.pth.tar ] || { echo "[$(date +%T)] MISSING checkpoint for $y" >> $LOG; return; }
  mkdir -p RGBT_workspace/results/LasHeR/$y; cp -n experiments/vipt/batch1_modality.yaml experiments/vipt/$y.yaml
  echo "[$(date +%T)] START $y" >> $LOG
  $PY RGBT_workspace/test_rgbt_mgpus.py --script_name vipt --dataset_name LasHeR --yaml_name $y --epoch 120 --mode parallel --threads 4 --seq_home /root/cvresearch/data/lasher/testingset > /root/cvresearch/B_eval_$y.log 2>&1
  echo "[$(date +%T)] END $y files=$(ls RGBT_workspace/results/LasHeR/$y | wc -l)" >> $LOG; }
for v in modality fedavg; do for r in 0020 0060 0100; do run B_conv_${v}_r${r}_thermal; done; done
echo "END $(date)" >> $LOG
