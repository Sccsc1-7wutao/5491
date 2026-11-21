#!/bin/bash
# Zero-aux (missing-modality) thermal evaluation of the two no-pure-RGB checkpoints on B. Runs only after B_daemon2 (batch8 thermal evals) has finished.
export CVR_ROOT=/root/cvresearch CVR_DATA=/root/cvresearch/data OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /root/cvresearch/ViPT
PY=/root/miniconda3/envs/sdstrack/bin/python
CK=output/checkpoints/train/vipt
LOG=/root/cvresearch/B_daemon4.log
while ! grep -q "^END" /root/cvresearch/B_daemon2.log 2>/dev/null; do sleep 60; done
echo "START $(date)" > $LOG
run() { y=$1
  while pgrep -f "[t]est_rgbt_mgpus" > /dev/null; do sleep 30; done
  mkdir -p RGBT_workspace/results/LasHeR/$y; cp -n experiments/vipt/batch1_modality.yaml experiments/vipt/$y.yaml
  echo "[$(date +%T)] START $y" >> $LOG
  $PY RGBT_workspace/test_rgbt_mgpus.py --script_name vipt --dataset_name LasHeR --yaml_name $y --epoch 120 --mode parallel --threads 4 --seq_home /root/cvresearch/data/lasher/testingset > /root/cvresearch/B_eval_$y.log 2>&1
  echo "[$(date +%T)] END $y files=$(ls RGBT_workspace/results/LasHeR/$y | wc -l)" >> $LOG; }
for pair in B_main_norgb_thermal:B_noaux_main_norgb B_norgb_thermal:B_noaux_fna_norgb; do src=${pair%%:*}; y=${pair##*:}; mkdir -p $CK/$y; cp $CK/$src/ViPTrack_ep0120.pth.tar $CK/$y/; done
export CVR_ZERO_AUX=1
run B_noaux_main_norgb
run B_noaux_fna_norgb
unset CVR_ZERO_AUX
echo "END $(date)" >> $LOG
