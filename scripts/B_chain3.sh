#!/bin/bash
export CVR_ROOT=/root/cvresearch CVR_DATA=/root/cvresearch/data OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /root/cvresearch/ViPT
PY=/root/miniconda3/envs/sdstrack/bin/python
CK=output/checkpoints/train/vipt
LOG=/root/cvresearch/B_chain3.log
while ! grep -q "^END" /root/cvresearch/B_chain2.log 2>/dev/null; do sleep 30; done
echo "START $(date)" > $LOG
run() { y=$1
  mkdir -p RGBT_workspace/results/LasHeR/$y; cp experiments/vipt/batch1_modality.yaml experiments/vipt/$y.yaml
  echo "[$(date +%T)] START $y" >> $LOG
  $PY RGBT_workspace/test_rgbt_mgpus.py --script_name vipt --dataset_name LasHeR --yaml_name $y --epoch 120 --mode parallel --threads 4 --seq_home /root/cvresearch/data/lasher/testingset > /root/cvresearch/B_eval_$y.log 2>&1
  echo "[$(date +%T)] END $y files=$(ls RGBT_workspace/results/LasHeR/$y | wc -l)" >> $LOG; }
run B_newfna
for c in 001 002 003; do run B_local_rgbt_$c; done
# ablation (aux+fusion) thermal: wait for its checkpoint on the shared disk
while [ ! -f /mnt/cvr_eval/ckpt_fa/DONE ]; do sleep 30; done
mkdir -p $CK/B_fa; cp /mnt/cvr_eval/ckpt_fa/ViPTrack_thermal_ep0120.pth.tar $CK/B_fa/ViPTrack_ep0120.pth.tar
run B_fa
echo "END $(date)" >> $LOG
