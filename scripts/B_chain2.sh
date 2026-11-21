#!/bin/bash
export CVR_ROOT=/root/cvresearch CVR_DATA=/root/cvresearch/data OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /root/cvresearch/ViPT
PY=/root/miniconda3/envs/sdstrack/bin/python
CK=output/checkpoints/train/vipt
echo "START $(date)" > /root/cvresearch/B_chain2.log
for c in rgbt_001 rgbt_002 rgbt_003; do
  y=B_fedper_$c
  mkdir -p $CK/$y RGBT_workspace/results/LasHeR/$y
  cp /mnt/cvr_eval/ckpt_fedper/ViPTrack_${c}_ep0120.pth.tar $CK/$y/ViPTrack_ep0120.pth.tar
  cp experiments/vipt/batch1_modality.yaml experiments/vipt/$y.yaml
  echo "[$(date +%T)] START $y" >> /root/cvresearch/B_chain2.log
  $PY RGBT_workspace/test_rgbt_mgpus.py --script_name vipt --dataset_name LasHeR --yaml_name $y --epoch 120 --mode parallel --threads 4 --seq_home /root/cvresearch/data/lasher/testingset > /root/cvresearch/B_eval_$y.log 2>&1
  echo "[$(date +%T)] END $y files=$(ls RGBT_workspace/results/LasHeR/$y | wc -l)" >> /root/cvresearch/B_chain2.log
done
echo "END $(date)" >> /root/cvresearch/B_chain2.log
